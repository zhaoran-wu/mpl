#pragma once

#include <Eigen/Core>
#include <string>
#include <vector>

namespace mpl {

/**
 * @brief Class to store all the camera parameters of image prymid
 * so we need to initialize it after the Config class
 *
 */
class CamData {
   public:
    static CamData& getInstance();

    bool readYamlFile(const std::string& file_path);

    CamData(const CamData&) = delete;
    CamData& operator=(const CamData&) = delete;

   private:
    CamData() = default;
    ~CamData() = default;

   public:
    // cameara patameters of image pyramid
    // e.g K of lvl 0 is K[0]
    std::vector<float> fx;
    std::vector<float> fy;
    std::vector<float> cx;
    std::vector<float> cy;

    std::vector<Eigen::Matrix3f> K;
    std::vector<Eigen::Matrix3f> K_inv;

    std::vector<int> width;
    std::vector<int> height;

    int lvls;  // num of image pyramid lvls
};

inline CamData& CamData::getInstance() {
    static CamData unique_config;
    return unique_config;
}

}  // namespace mpl