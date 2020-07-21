#include "synetic_image.h"
#include <algorithm>
#include <glad/glad.h>
// clang format off
#include <GLFW/glfw3.h>
// clang format on
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
namespace mpl {

SyneticImage::SyneticImage(const std::string project_path) {
    init();
    // load texture and mesh and upload to gpu
    mesh.read(project_path);
    // compile and link shader
    photometric_shader.read(project_path + "../shader/photometric.vs", project_path + "../shader/photometric.fs");
    depth_shader.read(project_path + "../shader/depth_vis.vs",
                      project_path + "../shader/depth_vis.fs");  // vis is for visualization
    normal_shader.read(project_path + "../shader/normal.vs", project_path + "../shader/normal.fs");

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
    // K that modified to fit opengl form
    Eigen::Matrix4f modified_K;
    modified_K << camera.fx, 0.0f, -camera.cx, 0.0f, 0.0f, camera.fy, -camera.cy, 0.0f, 0.0f, 0.0f, z_f + z_n,
        z_n * z_f, 0.0f, 0.0f, -1.0f, 0.0f;

    Eigen::Matrix4f NDC;
    NDC << 2.0f / camera.cols, 0.0f, 0.0f, -1.0f, 0.0f, 2.0f / camera.rows, 0.0f, -1.0f, 0.0f, 0.0f, 2.0f / (z_n - z_f),
        (z_n + z_f) / (z_n - z_f), 0.0f, 0.0f, 0.0f, 1.0f;

    projection = NDC * modified_K;
    zn_zf(0) = z_n;
    zn_zf(1) = z_f;

    extrinsic_T_c_cgl = Eigen::Isometry3f::Identity();  // cgl: opengl gl camera frame
                                                        // c: normal camera frame
    Eigen::Matrix3f extrinsic_R_c_cgl;
    extrinsic_R_c_cgl << 1, 0, 0, 0, -1, 0, 0, 0, -1;
    extrinsic_T_c_cgl.rotate(extrinsic_R_c_cgl);
}

void SyneticImage::setFrameBuffer() {
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffer);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
    // create a multisampled color attachment texture
    glGenRenderbuffers(1, &rbo_color);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo_color);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, camera.cols, camera.rows);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbo_color);
    // create a (also multisampled) renderbuffer object for depth and stencil
    // attachments
    glGenRenderbuffers(1, &rbo_depth_stencil);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo_depth_stencil);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH32F_STENCIL8, camera.cols, camera.rows);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo_depth_stencil);
    // check if a framebuffer is complete
    assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glClearColor(0.f, 0.f, 0.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // pxiel buffer object
    glGenBuffers(3, pbo_arr);  // pbo for batch read
    glGenBuffers(1, &pbo);     // pbo for read 1 buffer each times
}

cv::Mat SyneticImage::renderingAt(const Eigen::Isometry3f& pose, const RenderingMode mode) {
    view = (pose * extrinsic_T_c_cgl).inverse();

    glClearColor(0.f, 0.f, 0.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // rendering according to mode
    switch (mode) {
        case mpl::PHTOMETRIC: {
            rendering_photometric_image();
        } break;
        case mpl::DEPTH: {
            rendering_depth_image();

        } break;
        case mpl::NORMAL: {
            rendering_normal_image(pose);
        } break;
        default:
            break;
    }

    // copy from gpu to cpu
    if (mode == mpl::DEPTH) {
        cv::Mat result(camera.rows, camera.cols, CV_16UC4);
        cv::Mat result_single_channnel(camera.rows, camera.cols, CV_16UC1);

        glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
        glBufferData(GL_PIXEL_PACK_BUFFER, camera.cols * camera.rows * 4 * sizeof(ushort), 0, GL_STREAM_READ);
        glReadBuffer(GL_BACK);
        glReadPixels(0, 0, camera.cols, camera.rows, GL_RGBA, GL_UNSIGNED_SHORT, 0);

        result.data = (uchar*)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);

        glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

        cv::flip(result, result, 0);

        // transform to unit mm
        // (d/zf)* max(ushort)  = x ;  d_in_mm = x*z_f*1000/max(ushort)
        cv::cvtColor(result, result_single_channnel, CV_RGBA2GRAY);
        std::for_each(result_single_channnel.begin<ushort>(), result_single_channnel.end<ushort>(),
                      [=](ushort& depth) { depth *= (1.f / std::numeric_limits<ushort>::max()) * 1000 * z_f; });

        return result_single_channnel;
    } else {
        cv::Mat result(camera.rows, camera.cols, CV_8UC4);

        glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
        glBufferData(GL_PIXEL_PACK_BUFFER, camera.cols * camera.rows * 4 * sizeof(uchar), 0, GL_STREAM_READ);
        glReadBuffer(GL_BACK);

        glReadPixels(0, 0, camera.cols, camera.rows, GL_BGRA, GL_UNSIGNED_BYTE, 0);

        result.data = (uchar*)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);

        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

        cv::flip(result, result, 0);
        return result;
    }
}

inline void SyneticImage::rendering_photometric_image() {
    photometric_shader.active();
    photometric_shader.set_uniform_mat4("projection", projection);
    photometric_shader.set_uniform_mat_isometry("view", view);
    mesh.draw(photometric_shader);
}

inline void SyneticImage::rendering_depth_image() {
    depth_shader.active();
    depth_shader.set_uniform_mat4("projection", projection);
    depth_shader.set_uniform_mat_isometry("view", view);
    depth_shader.set_uniform_vec2("zn_zf", zn_zf);
    mesh.draw(depth_shader);
}

inline void SyneticImage::rendering_normal_image(const Eigen::Isometry3f& pose) {
    normal_shader.active();
    normal_shader.set_uniform_mat4("projection", projection);
    normal_shader.set_uniform_mat_isometry("view", view);
    normal_shader.set_uniform_mat3("normal_R", pose.inverse().rotation().matrix());
    mesh.draw(normal_shader);
}

void SyneticImage::set_start_pose(const Eigen::Isometry3f& pose) {
    this->start_T_w_c0 = Sophus::SE3f(pose.rotation(), pose.translation());
}

std::vector<cv::Mat> SyneticImage::renderingAt(const Eigen::Isometry3f& pose) {
    view = (pose * extrinsic_T_c_cgl).inverse();

    glClearColor(0.f, 0.f, 0.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    std::vector<cv::Mat> results;

    // rendering photometric image
    rendering_photometric_image();
    // copy from gpu to cpu
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_arr[0]);
    glBufferData(GL_PIXEL_PACK_BUFFER, camera.cols * camera.rows * 4 * sizeof(uchar), 0, GL_STREAM_READ);
    glReadBuffer(GL_BACK);
    // use pbo, glReadPixels will return immedeatly, which allow
    // asynchronization
    glReadPixels(0, 0, camera.cols, camera.rows, GL_BGRA, GL_UNSIGNED_BYTE, 0);

    // rendring depth image
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    rendering_depth_image();

    // copy from gpu to cpu

    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_arr[1]);
    glBufferData(GL_PIXEL_PACK_BUFFER, camera.cols * camera.rows * 4 * sizeof(ushort), 0, GL_STREAM_READ);
    glReadBuffer(GL_BACK);
    glReadPixels(0, 0, camera.cols, camera.rows, GL_RGBA, GL_UNSIGNED_SHORT, 0);

    // when read depth data ,we can map and process photometric data
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_arr[0]);
    cv::Mat photometric(camera.rows, camera.cols, CV_8UC4);
    photometric.data = (uchar*)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    cv::flip(photometric, photometric, 0);
    results.push_back(photometric);

    // rendering normal image
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    rendering_normal_image(pose);

    // copy from gpu to cpu

    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_arr[2]);
    glBufferData(GL_PIXEL_PACK_BUFFER, camera.cols * camera.rows * 4 * sizeof(uchar), 0, GL_STREAM_READ);
    glReadBuffer(GL_BACK);
    glReadPixels(0, 0, camera.cols, camera.rows, GL_BGRA, GL_UNSIGNED_BYTE, 0);

    // when read normal map data, we can map and process the result from depth
    // map
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_arr[1]);
    cv::Mat depth(camera.rows, camera.cols, CV_16UC4);
    cv::Mat depth_single_channnel(camera.rows, camera.cols, CV_16UC1);
    depth.data = (uchar*)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    cv::flip(depth, depth, 0);

    // transform to unit mm
    // (d/zf)* max(ushort)  = x ;  d_in_mm = x*z_f*1000/max(ushort)
    cv::cvtColor(depth, depth_single_channnel, CV_RGBA2GRAY);
    std::for_each(depth_single_channnel.begin<ushort>(), depth_single_channnel.end<ushort>(),
                  [=](ushort& depth) { depth *= (1.f * 1000 * z_f / std::numeric_limits<ushort>::max()); });

    results.push_back(depth_single_channnel);

    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_arr[2]);
    cv::Mat normal(camera.rows, camera.cols, CV_8UC4);
    normal.data = (uchar*)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
    cv::flip(normal, normal, 0);

    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    results.push_back(normal);

    return results;
}

std::vector<cv::Mat> SyneticImage::renderingAt(const Sophus::SE3f& pose) {
    Eigen::Isometry3f T_w_c = static_cast<Eigen::Isometry3f>((start_T_w_c0 * pose).matrix());
    return renderingAt(T_w_c);
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