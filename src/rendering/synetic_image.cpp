#include "synetic_image.h"
#include <iostream>
namespace mpl {

SyneticImage::SyneticImage(const std::string project_path) {
  init();
  // load texture and mesh and upload to gpu
  mesh.read(project_path);
  // compile and link shader
  photometric_shader.read(project_path + "../shader/photometric.vs",
                          project_path + "../shader/photometric.fs");
  depth_shader.read(
      project_path + "../shader/depth_vis.vs",
      project_path + "../shader/depth_vis.fs");  // vis is for visualization
  normal_shader.read(project_path + "../shader/normal.vs",
                     project_path + "../shader/normal.fs");

  std::string config_file = project_path + "/config_camera.yaml";
  util::exists(config_file);
  util::SensorDataMap sensor_map = util::readSensorDataFromYaml(config_file);
  camera = sensor_map.begin()->second;
  preComputeParam();
  setFrameBuffer();
}

void SyneticImage::init() {
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_VISIBLE, false);

  window = glfwCreateWindow(1241, 376, "test_synetic_image", NULL, NULL);
  if (window == NULL) {
    std::cout << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
  }
  glfwMakeContextCurrent(window);
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cout << "Failed to initialize GLAD" << std::endl;
  }

  // configure global opengl state
  // -----------------------------
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
}

void SyneticImage::preComputeParam() {
  // use the intrinsic parameters
  constexpr float z_n = 1;
  constexpr float z_f = 40;

  // K that modified to fit opengl form
  Eigen::Matrix4f modified_K;
  modified_K << camera.fx, 0.0f, -camera.cx, 0.0f, 0.0f, camera.fy, -camera.cy,
      0.0f, 0.0f, 0.0f, z_f + z_n, z_n * z_f, 0.0f, 0.0f, -1.0f, 0.0f;

  Eigen::Matrix4f NDC;
  NDC << 2.0f / camera.cols, 0.0f, 0.0f, -1.0f, 0.0f, 2.0f / camera.rows, 0.0f,
      -1.0f, 0.0f, 0.0f, 2.0f / (z_n - z_f), (z_n + z_f) / (z_n - z_f), 0.0f,
      0.0f, 0.0f, 1.0f;

  projection = NDC * modified_K;
  zn_zf(0) = z_n;
  zn_zf(1) = z_f;

  Tb_gl = Eigen::Isometry3f::Identity();
  Eigen::Matrix3f Rb_gl;
  Rb_gl << 1, 0, 0, 0, -1, 0, 0, 0, -1;
  Tb_gl.rotate(Rb_gl);
}

void SyneticImage::setFrameBuffer() {
  glGenFramebuffers(1, &framebuffer);
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffer);
  // create a multisampled color attachment texture
  glGenRenderbuffers(1, &rbo_color);
  glBindRenderbuffer(GL_RENDERBUFFER, rbo_color);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, camera.cols, camera.rows);
  glBindRenderbuffer(GL_RENDERBUFFER, 0);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                            GL_RENDERBUFFER, rbo_color);
  // create a (also multisampled) renderbuffer object for depth and stencil
  // attachments
  glGenRenderbuffers(1, &rbo_depth_stencil);
  glBindRenderbuffer(GL_RENDERBUFFER, rbo_depth_stencil);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH32F_STENCIL8, camera.cols,
                        camera.rows);
  glBindRenderbuffer(GL_RENDERBUFFER, 0);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                            GL_RENDERBUFFER, rbo_depth_stencil);
  // check if a framebuffer is complete
  assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
  glClearColor(0.f, 0.f, 0.f, 1.f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

cv::Mat SyneticImage::renderingAt(const Eigen::Isometry3f& pose,
                                  const RenderingMode mode) {
  view = (pose * Tb_gl).inverse();
  // todo switch

  glClearColor(0.f, 0.f, 0.f, 1.f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  switch (mode) {
    case mpl::PHTOMETRIC: {
      photometric_shader.active();
      photometric_shader.set_uniform_mat4("projection", projection);
      photometric_shader.set_uniform_mat4("view", view);
      mesh.draw(photometric_shader);
    } break;
    case mpl::DEPTH: {
      depth_shader.active();
      depth_shader.set_uniform_mat4("projection", projection);
      depth_shader.set_uniform_mat4("view", view);
      depth_shader.set_uniform_vec2("zn_zf", zn_zf);
      mesh.draw(depth_shader);
    } break;
    case mpl::NORMAL: {
      normal_shader.active();
      normal_shader.set_uniform_mat4("projection", projection);
      normal_shader.set_uniform_mat4("view", view);
      normal_shader.set_uniform_mat3("normal_R", pose.inverse().rotation());
      mesh.draw(normal_shader);
    } break;
    default:
      break;
  }

  // Copy back in the given Eigen matrices
  if (mode == mpl::DEPTH) {
    cv::Mat result(camera.rows, camera.cols, CV_8UC4);
    glReadPixels(0, 0, camera.cols, camera.rows, GL_BGRA, GL_UNSIGNED_BYTE,
                 result.data);

    cv::flip(result, result, 0);
    return result;
  } else {
    cv::Mat result(camera.rows, camera.cols, CV_8UC4);
    glReadPixels(0, 0, camera.cols, camera.rows, GL_BGRA, GL_UNSIGNED_BYTE,
                 result.data);
    cv::flip(result, result, 0);
    return result;
  }
}

void SyneticImage::shut() {
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
  glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glDeleteRenderbuffers(1, &rbo_color);
  glDeleteFramebuffers(1, &framebuffer);
  glDeleteRenderbuffers(1, &rbo_depth_stencil);
  glfwTerminate();
}

}  // namespace mpl