#include "mesh.h"
#include "shader.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "../util/trajectory.h"
#include "../util/utility.h"
#include <iostream>
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);

// settings
const unsigned int SCR_WIDTH = 1241;
const unsigned int SCR_HEIGHT = 376;

// camera
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;
using namespace mpl;
int main() {
  // glfw: initialize and configure
  // ------------------------------
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
  glfwWindowHint(
      GLFW_OPENGL_FORWARD_COMPAT,
      GL_TRUE);  // uncomment this statement to fix compilation on OS X
#endif

  // glfw window creation
  // --------------------
  GLFWwindow* window =
      glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "test_synetic_image", NULL, NULL);
  if (window == NULL) {
    std::cout << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

  // tell GLFW to capture our mouse
  glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

  // glad: load all OpenGL function pointers
  // ---------------------------------------
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cout << "Failed to initialize GLAD" << std::endl;
    return -1;
  }

  // configure global opengl state
  // -----------------------------
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  // build and compile shaders
  // -------------------------
  std::string vs_file_path = "/home/zhaoran/thesis_ws/mpl/shader/photometric.vs";
  std::string fs_file_path = "/home/zhaoran/thesis_ws/mpl/shader/photometric.fs";
  std::string vs_file_path2 = "/home/zhaoran/thesis_ws/mpl/shader/depth_vis.vs";
  std::string fs_file_path2 = "/home/zhaoran/thesis_ws/mpl/shader/depth_vis.fs";
  std::string vs_file_path3 = "/home/zhaoran/thesis_ws/mpl/shader/normal.vs";
  std::string fs_file_path3 = "/home/zhaoran/thesis_ws/mpl/shader/normal.fs";
  Shader photometirc_shader;
  photometirc_shader.read(vs_file_path.c_str(), fs_file_path.c_str());
  Shader depth_shader;
  depth_shader.read(vs_file_path2.c_str(), fs_file_path2.c_str());
  Shader normal_shader;
  depth_shader.read(vs_file_path3.c_str(), fs_file_path3.c_str());

  // load models
  // -----------
  std::string ply_file = "/home/zhaoran/dataset/mrt/uniqueVertexMesh.ply";
  std::string texture_file = "/home/zhaoran/dataset/mrt/texture.png";
  Mesh ourModel;
  ourModel.read("/home/zhaoran/thesis_ws/mpl/project/");

  // read the pose file
  std::string pose_file = "/home/zhaoran/dataset/mrt/poses_cam_gt.txt";
  trajectory_io::Trajectory pose;
  util::exists(pose_file);
  pose.read(pose_file, trajectory_io::Trajectory::FORMAT_MAT);

  Eigen::Isometry3f first_pose = pose.atIndex(0);
  Eigen::Matrix3f normal_R = first_pose.rotation().inverse();
  // try to build view matrix of the first frame
  // todo view correction because of off-center
  Eigen::Isometry3f Tb_gl = Eigen::Isometry3f::Identity();
  Eigen::Matrix3f Rb_gl;
  Rb_gl << 1, 0, 0, 0, -1, 0, 0, 0, -1;

  Tb_gl.rotate(Rb_gl);

  Eigen::Isometry3f view_first = (first_pose * Tb_gl).inverse();

  // use the intrinsic parameters
  const float FX = 718.85f, FY = 718.85f, CX = 607.2f, CY = 185.2f;
  const float z_n = 0.1f;
  const float z_f = 100;

  Eigen::Matrix4f K;
  K << FX, 0, -CX, 0, 0, FY, -CY, 0, 0, 0, z_f + z_n, z_n * z_f, 0, 0, -1, 0;

  Eigen::Matrix4f NDC;
  // todo use more precise formular
  NDC << 2.f / 1241, 0, 0, -1, 0, 2.f / 376, 0, -1, 0, 0, 2 / (z_n - z_f),
      (z_n + z_f) / (z_n - z_f), 0, 0, 0, 1;

  Eigen::Matrix4f P = NDC * K;

  Eigen::Vector2f zn_zf(z_n, z_f);

  bool useDepthShader = false;
  bool useNormalMap = true;
  while (!glfwWindowShouldClose(window)) {
    // per-frame time logic
    // --------------------
    float currentFrame = glfwGetTime();
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;

    // input
    // -----
    processInput(window);

    // render
    // ------
    glClearColor(0.1f, 0.1f, 0.05f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // don't forget to enable shader before setting uniforms
         if (!useDepthShader) {
          photometirc_shader.active();
          photometirc_shader.set_uniform_mat4("projection", P);
          photometirc_shader.set_uniform_mat4("view", view_first);
          useDepthShader = true;

        } else {
          depth_shader.active();
          depth_shader.set_uniform_vec2("zn_zf", zn_zf);
          depth_shader.set_uniform_mat4("projection", P);
          depth_shader.set_uniform_mat4("view", view_first);
          useDepthShader = false;
        } 
/*     normal_shader.active();
    normal_shader.set_uniform_mat4("projection", P);
    normal_shader.set_uniform_mat4("view", view_first);
    normal_shader.set_uniform_mat3("normal_R",normal_R);
    useDepthShader = false */;

    // view/projection transformations
    // glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom),
    // (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f); glm::mat4 view =
    // camera.GetViewMatrix();
    // photometirc_shader.setMat4("projection",projection);
    // photometirc_shader.setMat4("view", view);
    // render the loaded model
    // glBindFramebuffer(GL_FRAMEBUFFER, ourModel.framebuffer)

    ourModel.draw(photometirc_shader);

    // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved
    // etc.)
    // -------------------------------------------------------------------------------
    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  // glfw: terminate, clearing all previously allocated GLFW resources.
  // ------------------------------------------------------------------
  glfwTerminate();
  return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this
// frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback
// function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
  // make sure the viewport matches the new window dimensions; note that width
  // and height will be significantly larger than specified on retina displays.
  glViewport(0, 0, width, height);
}
