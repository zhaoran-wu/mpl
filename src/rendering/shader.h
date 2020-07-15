#ifndef SHADER_H
#define SHADER_H

#include <Eigen/Geometry>
#include <glad/glad.h>
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
    void active() const {
        glUseProgram(ID);
    }
    template <typename T>
    void set_uniform_mat4(const std::string& name, const T& mat) const {
        glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, mat.data());
    }
    template <typename T>
    void set_uniform_mat3(const std::string& name, const T& mat) const {
        glUniformMatrix3fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, mat.data());
    }
    template <typename T>
    void set_uniform_vec2(const std::string& name, const T& vec) const {
        glUniform2fv(glGetUniformLocation(ID, name.c_str()), 1, vec.data());
    }

   private:
    void checkCompileErrors(GLuint shader, std::string type);
};
#endif
}  // namespace mpl