#ifndef MPL_SYNETIC_IMAGE
#define MPL_SYNETIC_IMAGE
#include "../util/utility.h"
#include "mesh.h"
#include "shader.h"
#include <GLFW/glfw3.h>
#include <glad/glad.h>

#include <opencv2/core.hpp>
#include <string>
namespace mpl {

enum RenderingMode { PHTOMETRIC, DEPTH, NORMAL };
class SyneticImage {
 public:
  SyneticImage(const std::string project_path);
  cv::Mat renderingAt(const Eigen::Isometry3f& pose, const RenderingMode mode);
  void shut();

 private:
  /**
   * @brief create opengl context
   *
   */
  void init();
  /**
   * @brief pre-compute some parameters, which will be set as uniform parameters
   * for shader
   *
   */
  void preComputeParam();

  void setFrameBuffer();


  Mesh mesh;  // all the ply and texture data

  Shader photometric_shader;
  Shader depth_shader;
  Shader normal_shader;

  unsigned int framebuffer;
  unsigned int rbo_color;
  unsigned int rbo_depth_stencil;

  // uniform parameters for opengl
  Eigen::Matrix4f projection;  //! projection matrix of opengl
  Eigen::Isometry3f view;      //! view matix of opengl, correspond to Tgl_w
  Eigen::Vector2f zn_zf;       //! z_near and z_far;
  Eigen::Matrix3f Rb_w;        // ! world to body frame
  Eigen::Isometry3f Tb_gl;     // ! opengl camera to body frame

  // window
  GLFWwindow* window;
  // sensor data
  util::SensorData camera;
};

}  // namespace mpl
#endif  //! MPL_SYNETIC_IMAGE