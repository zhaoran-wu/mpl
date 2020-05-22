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
    /**
     * @brief return rendering image from model and texture at given pose, and
     * mode
     *
     * @param pose T_b0_bfn,the tranform from curr body frame bfn to body frame
     * at world frame identity b0 , T_b0_bfn =  T_b0_bf0  * T_bf0_bfn
     * = T_b0_bf0 * ( extrin_T_b_c    T_c0_cn * extrin_T_c_b )
     * @param mode support color image ,depth image(mm, store in ushort), and normal image
     * @return cv::Mat
     */
    cv::Mat renderingAt(const Eigen::Isometry3f& pose, const RenderingMode mode);
    void shut();

   private:
    /**
     * @brief create opengl context
     *
     */
    void init();
    /**
     * @brief pre-compute some parameters, which will be set as uniform
     * parameters for shader
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

    GLuint pbo;

    // uniform parameters for opengl
    Eigen::Matrix4f projection;          //! projection matrix of opengl
    Eigen::Isometry3f view;              //! view matix of opengl, correspond to Tgl_w
    Eigen::Vector2f zn_zf;               //! z_near and z_far;
    Eigen::Matrix3f Rb_w;                // ! world to body frame
    Eigen::Isometry3f extrinsic_T_b_gl;  // ! opengl camera to body frame

    const float z_n = 1.f;
    const float z_f = 60.f;

    // window
    GLFWwindow* window;
    // sensor data
    util::SensorData camera;
};

}  // namespace mpl
#endif  //! MPL_SYNETIC_IMAGE