#ifndef MESH_H
#define MESH_H

#include <string>
#include <vector>
#include "../util/ply.h"
#include "../util/utility.h"
namespace mpl {
class Shader;


/**
 * @brief a class to read und upload ply/texture file and upload it to gpu
 *  and hold the correspond opengl object
 */

class Mesh {
 public:
  Mesh() = default;
  PlyFile ply_file;
  unsigned int texture_id;
  unsigned int VAO;

  void read(const std::string project_path);

  // render the mesh
  void draw(Shader shader);

 private:
  unsigned int VBO, EBO;
  //todo launch ply using binary file to speed up
  void loadPlyData(const std::string ply_file_path);

  void loadTextureImage(const std::string texture_file_path);

  void uploadToGpu();

  util::SensorData camera_config; 


};
#endif
}  //! namespace mpl