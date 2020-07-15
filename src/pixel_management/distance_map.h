#pragma once

#include <Eigen/Core>

#include <memory>
#include <vector>

#include "frame.h"
#include "opencv2/core.hpp"
#include <unordered_map>

namespace mpl {
/**
 * @brief  distance map of candidate pixle on image using square distance
 */
class Candidate;
class DistanceMap {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    DistanceMap();
    ~DistanceMap();

    // access distance of the pixel u, v in the map
    float dist(int x, int y) const;

    int get_num_obstacles() const;

    // compute distance map of active candidate on all other frames project on
    // given frame
    void compute(std::unordered_map<Frame::ptr, std::vector<Candidate>>& candidate_map, const Frame::ptr frame);

    // adds new pixel to map and updates distance
    void add(const Eigen::Vector2f point_in_newst_KF);

    // draws the map
    cv::Mat get_distance_map_for_visualization(bool normalize);

   private:
    // calculate distance for all pixel on a square edge with radius
    bool update_pixel_on_square_edge(int x, int y, int radius);

   private:
    int w, h;  // image size

    float* dist_map;             // distance map
    Eigen::Vector2i* obstacles;  // obstacle in the map with dist = 0
    bool* require_update;        // flag to stop update the dist map
    int num_obstacles = 0;       // number of obstacles in the map

    CamData* cam;
};
}  // namespace mpl