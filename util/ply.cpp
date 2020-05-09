#include "ply.h"
#include "logging.h"
#include <Eigen/Geometry>
#include <boost/algorithm/string.hpp>
#include <fstream>

namespace mpl {
std::vector<std::string> StringSplit(const std::string& str,
                                     const std::string& delim) {
  std::vector<std::string> elems;
  boost::split(elems, str, boost::is_any_of(delim), boost::token_compress_on);
  return elems;
}

PlyFile readPlyFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  CHECK(file.is_open()) << path;

  PlyFile ply_file;

  std::string line;

  // The index of the property for ASCII PLY files.
  int X_index = -1;
  int Y_index = -1;
  int Z_index = -1;
  int NX_index = -1;
  int NY_index = -1;
  int NZ_index = -1;
  int S_index = -1;
  int T_index = -1;
  // The position in number of bytes of the property for binary PLY files.
  int X_byte_pos = -1;
  int Y_byte_pos = -1;
  int Z_byte_pos = -1;
  int NX_byte_pos = -1;
  int NY_byte_pos = -1;
  int NZ_byte_pos = -1;
  int S_byte_pos = -1;
  int T_byte_pos = -1;

  bool is_binary = false;
  bool is_vertex_section = false;
  size_t num_bytes_per_line = 0;
  size_t num_vertices = 0;
  size_t num_faces = 0;

  int index = 0;
  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;
    }

    if (line == "end_header") {
      break;
    }

    if (line.size() >= 6 && line.substr(0, 6) == "format") {
      if (line == "format ascii 1.0") {
        is_binary = false;
      } else if (line == "format binary_little_endian 1.0") {
        is_binary = true;
      }
    }

    const std::vector<std::string> line_elems = StringSplit(line, " ");

    if (line_elems.size() >= 3 && line_elems[0] == "element") {
      if (line_elems[1] == "vertex") {
        is_vertex_section = true;
        num_vertices = std::stoll(line_elems[2]);
      } else if (line_elems[1] == "face") {
        num_faces = std::stoll(line_elems[2]);
        is_vertex_section = false;
      } else {
        LOG(FATAL) << "ply file format not supported";
      }
    }

    if (line_elems.size() >= 3 && line_elems[0] == "property" &&
        is_vertex_section) {
      CHECK(line_elems[1] == "float" || line_elems[1] == "float32")
          << "PLY import only supports the float and float32 data types";

      if (line == "property float x" || line == "property float32 x") {
        X_index = index;
        X_byte_pos = num_bytes_per_line;
      } else if (line == "property float y" || line == "property float32 y") {
        Y_index = index;
        Y_byte_pos = num_bytes_per_line;
      } else if (line == "property float z" || line == "property float32 z") {
        Z_index = index;
        Z_byte_pos = num_bytes_per_line;
      } else if (line == "property float nx" || line == "property float32 nx") {
        NX_index = index;
        NX_byte_pos = num_bytes_per_line;
      } else if (line == "property float ny" || line == "property float32 ny") {
        NY_index = index;
        NY_byte_pos = num_bytes_per_line;
      } else if (line == "property float nz" || line == "property float32 nz") {
        NZ_index = index;
        NZ_byte_pos = num_bytes_per_line;
      } else if (line == "property float s" || line == "property float32 s") {
        S_index = index;
        S_byte_pos = num_bytes_per_line;
      } else if (line == "property float t" || line == "property float32 t") {
        T_index = index;
        T_byte_pos = num_bytes_per_line;
      } else {
        LOG(FATAL) << "Invalid data type: " << line_elems[1];
      }
      index += 1;
      num_bytes_per_line += 4;
    }
  }

  ply_file.vertices.reserve(num_vertices);
  ply_file.indices.reserve(num_faces*3);
  if (is_binary) {
    std::vector<char> vtx_buffer(num_bytes_per_line);
    int num_bytes_per_line_face = sizeof(uint8_t) + sizeof(size_t) * 3;
    std::vector<char> face_buffer(num_bytes_per_line_face);

    for (size_t i = 0; i < num_vertices + num_faces; ++i) {
      if (i < num_vertices) {
        file.read(vtx_buffer.data(), num_bytes_per_line);
        Vertex vtx;

        vtx.position[0] = *reinterpret_cast<float*>(&vtx_buffer[X_byte_pos]);
        vtx.position[1] = *reinterpret_cast<float*>(&vtx_buffer[Y_byte_pos]);
        vtx.position[2] = *reinterpret_cast<float*>(&vtx_buffer[Z_byte_pos]);

        vtx.normal[0] = *reinterpret_cast<float*>(&vtx_buffer[NX_byte_pos]);
        vtx.normal[1] = *reinterpret_cast<float*>(&vtx_buffer[NY_byte_pos]);
        vtx.normal[2] = *reinterpret_cast<float*>(&vtx_buffer[NZ_byte_pos]);

        vtx.texCoords[0] = *reinterpret_cast<float*>(&vtx_buffer[S_byte_pos]);
        vtx.texCoords[1] = *reinterpret_cast<float*>(&vtx_buffer[T_byte_pos]);

        ply_file.vertices.push_back(vtx);
      } else {
        file.read(face_buffer.data(), num_bytes_per_line_face);
        unsigned int vertex_idx1 = *reinterpret_cast<size_t*>(&face_buffer[sizeof(char)]);
        unsigned int vertex_idx2 =
           *reinterpret_cast<size_t*>(&face_buffer[sizeof(char) + sizeof(size_t)]);
        unsigned int vertex_idx3 = *reinterpret_cast<size_t*>(
            &face_buffer[sizeof(char) + 2 * sizeof(size_t)]);

        ply_file.indices.push_back(vertex_idx1);
        ply_file.indices.push_back(vertex_idx2);
        ply_file.indices.push_back(vertex_idx3);
        
      }
    }
  } else {
    for (size_t i = 0; i < num_vertices + num_faces; ++i) {
      std::getline(file, line);
      std::stringstream line_stream(line);
      std::string item;
      std::vector<std::string> items;
      while (!line_stream.eof()) {
        std::getline(line_stream, item, ' ');
        items.push_back(item);
      }
      if (i < num_vertices) {
        Vertex vtx;

        vtx.position[0] = std::stold(items.at(X_index));
        vtx.position[1] = std::stold(items.at(Y_index));
        vtx.position[2] = std::stold(items.at(Z_index));

        vtx.normal[0] = std::stold(items.at(NX_index));
        vtx.normal[1] = std::stold(items.at(NY_index));
        vtx.normal[2] = std::stold(items.at(NZ_index));
        vtx.texCoords[0] = std::stold(items.at(S_index));
        vtx.texCoords[1] = std::stold(items.at(T_index));

        ply_file.vertices.push_back(vtx);
      } else {
        unsigned int vertex_idx1 = std::stoi(items.at(1));
        unsigned int vertex_idx2 = std::stoi(items.at(2));
        unsigned int vertex_idx3 = std::stoi(items.at(3));
        ply_file.indices.push_back(vertex_idx1);
        ply_file.indices.push_back(vertex_idx2);
        ply_file.indices.push_back(vertex_idx3);
      }
    }
  }
  return ply_file;
}

void writeBinaryPlyFile(const std::string& path, const PlyFile& ply_file) {
  std::fstream text_file(path, std::ios::out);
  CHECK(text_file.is_open());

  text_file << "ply" << std::endl;
  text_file << "format binary_little_endian 1.0" << std::endl;
  text_file << "element vertex " << ply_file.vertices.size() << std::endl;
  text_file << "property float x" << std::endl;
  text_file << "property float y" << std::endl;
  text_file << "property float z" << std::endl;
  text_file << "property float nx" << std::endl;
  text_file << "property float ny" << std::endl;
  text_file << "property float nz" << std::endl;
  text_file << "property float s" << std::endl;
  text_file << "property float t" << std::endl;
  text_file << "element face " << ply_file.indices.size()/3 << std::endl;
  text_file << "property list uchar int vertex_index" << std::endl;
  text_file << "end_header" << std::endl;
  text_file.close();

  std::fstream binary_file(path,
                           std::ios::out | std::ios::binary | std::ios::app);
  CHECK(binary_file.is_open()) << path;

  for (const auto& vtx : ply_file.vertices) {
    binary_file.write(reinterpret_cast<const char*>(&vtx.position[0]), sizeof(float));
    binary_file.write(reinterpret_cast<const char*>(&vtx.position[1]), sizeof(float));
    binary_file.write(reinterpret_cast<const char*>(&vtx.position[2]), sizeof(float));
    binary_file.write(reinterpret_cast<const char*>(&vtx.normal[0]), sizeof(float));
    binary_file.write(reinterpret_cast<const char*>(&vtx.normal[1]), sizeof(float));
    binary_file.write(reinterpret_cast<const char*>(&vtx.normal[2]), sizeof(float));
    binary_file.write(reinterpret_cast<const char*>(&vtx.texCoords[0]), sizeof(float));
    binary_file.write(reinterpret_cast<const char*>(&vtx.texCoords[1]), sizeof(float));
  }

  for (size_t i = 0; i < ply_file.indices.size(); i +=3 ) {
    
    const uint8_t numVtxPerFace = 3;
    binary_file.write(reinterpret_cast<const char*>(&numVtxPerFace),
                      sizeof(uint8_t));
    binary_file.write(reinterpret_cast<const char*>(&ply_file.indices[i]),
                      sizeof(size_t));
    binary_file.write(reinterpret_cast<const char*>(&ply_file.indices[i+1]),
                      sizeof(size_t));
    binary_file.write(reinterpret_cast<const char*>(&ply_file.indices[i+2]),
                      sizeof(size_t));
  }

  binary_file.close();
}


void convertNormal(PlyFile& ply_file){
  for(size_t i = 0; i < ply_file.indices.size(); i +=3 ){
    auto& v1 = ply_file.vertices[ply_file.indices[i]];
    auto& v2 = ply_file.vertices[ply_file.indices[i+1]];
    auto& v3 = ply_file.vertices[ply_file.indices[i+2]];

    const auto& p1 = v1.position;
    const auto& p2 = v2.position;
    const auto& p3 = v3.position;

    Eigen::Vector3f p12(p2[0]- p1[0],p2[1]-p1[1],p2[2]-p1[2]);
    Eigen::Vector3f p13(p3[0]- p1[0],p3[1]-p1[1],p3[2]-p1[2]);

    Eigen::Vector3f new_normal = p12.cross(p13).normalized();

    v1.normal[0] = new_normal(0);
    v1.normal[1] = new_normal(1);
    v1.normal[2] = new_normal(2);
    v2.normal[0] = new_normal(0);
    v2.normal[1] = new_normal(0);
    v2.normal[2] = new_normal(1);
    v3.normal[0] = new_normal(0);
    v3.normal[1] = new_normal(0);
    v3.normal[2] = new_normal(1);

  }
  LOG(INFO) <<" normal convert finish " ; 
}

}  // namespace mpl
