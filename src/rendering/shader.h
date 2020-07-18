#ifndef SHADER_H
#define SHADER_H

#include <Eigen/Geometry>
#include <string>

namespace mpl {

/**
 * @brief class to read vertex shader, fragement shader, compile and link to build a shader
 */
class Shader {
   public:
    unsigned int ID;
    Shader() = default;
    void read(const std::string vertexPath,
              const std::string fragmentPath);  //! put the shader folder in the root folder of project folder
    void active() const;
    void set_uniform_mat4(const std::string& name, const Eigen::Matrix4f& mat) const;
    void set_uniform_mat3(const std::string& name, const Eigen::Matrix3f& mat) const;
    void set_uniform_mat_isometry(const std::string& name, const Eigen::Isometry3f& mat) const;
    void set_uniform_vec2(const std::string& name, const Eigen::Vector2f& vec) const;

   private:
    void checkCompileErrors(uint shader, std::string type);
};
#endif
}  // namespace mpl