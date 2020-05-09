#ifndef MPL_UTL_PLY
#define MPL_UTL_PLY

#include <string>
#include <vector>

namespace mpl {

struct Vertex {
  float position[3];
  float normal[3];
  float texCoords[2];
};

struct PlyFile {
  std::vector<Vertex> vertices;
  std::vector<unsigned int> indices;
};

// Read PLY point cloud from text or binary file.
PlyFile readPlyFile(const std::string& path);

// Write PLY mesh to text or binary file.
void writeBinaryPlyFile(const std::string& path, const PlyFile& ply_file);

// compute normal information with vertex, ignore original normal information
void convertNormal(PlyFile& ply_file);

}//!namespace : mpl

#endif 