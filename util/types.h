#pragma once

#include <map>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <optional>

namespace util {

template<typename T>
using eigenStdVector = std::vector<T, Eigen::aligned_allocator<T>>;

struct SensorData {
    std::string id;
    Eigen::Isometry3f extrinsics = Eigen::Isometry3f::Identity();
    float fx, fy, cx, cy;
    int rows, cols;
    bool disabled = false;
    Eigen::Vector3f cOffset = Eigen::Vector3f::Zero();

//    float sigma;
//    float projection_type;
//    int fill_radius;
//    float intr_scale;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

using SensorDataMap = std::map<std::string, SensorData>;

} // namespace util