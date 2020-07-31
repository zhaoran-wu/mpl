
#include <eigen3/Eigen/Core>
#include <glad/glad.h>  // holds all OpenGL type declarations
#include <glog/logging.h>
#include <opencv2/highgui.hpp>

#include <fstream>
#include <iostream>
#include <sstream>

#include "mesh.h"
#include "shader.h"

namespace mpl {

void Mesh::read(const std::string project_path) {
    loadPlyData(project_path + "out.ply");
    loadTextureImage(project_path + "texture.png");
    util::SensorDataMap sensor_data = util::readSensorDataFromYaml(project_path + "config_camera.yaml");
    camera_config = sensor_data.begin()->second;
    uploadToGpu();
}

// render the mesh
void Mesh::draw(Shader shader) {
    glActiveTexture(GL_TEXTURE0);
    // now set the sampler to the correct texture unit
    glUniform1i(glGetUniformLocation(shader.ID, "texture1"), 0);
    glBindTexture(GL_TEXTURE_2D, texture_id);

    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, ply_file.indices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void Mesh::loadPlyData(const std::string ply_file_path) {
    LOG(INFO) << " loading ply file ";
    ply_file = readPlyFile(ply_file_path);
    LOG(INFO) << " ply file loaded";
}

void Mesh::loadTextureImage(const std::string texture_file_path) {
    LOG(INFO) << "loading texture file";
    cv::Mat im_bgr = cv::imread(texture_file_path);
    cv::flip(im_bgr, im_bgr, 0);

    // todo use GL_GL_LUMINANCE
    if (im_bgr.data) {
        glGenTextures(1, &texture_id);
        glBindTexture(GL_TEXTURE_2D, texture_id);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, im_bgr.cols, im_bgr.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, im_bgr.data);

        LOG(INFO) << "texture loaded";
    } else {
        LOG(FATAL) << "Falid to load texture ";
    }
    // set wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    // set filtring parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_LINEAR);

    glGenerateMipmap(GL_TEXTURE_2D);
}

void Mesh::uploadToGpu() {
    // create buffers/arrays
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);
    // load data into vertex buffers
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    glBufferData(GL_ARRAY_BUFFER, ply_file.vertices.size() * sizeof(Vertex), &ply_file.vertices[0], GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, ply_file.indices.size() * sizeof(unsigned int), &ply_file.indices[0],
                 GL_STATIC_DRAW);

    // set the vertex attribute pointers
    // vertex Positions
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    // vertex normals
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
    // vertex texture coords
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, texCoords));
    glBindVertexArray(0);
}
}  // namespace mpl