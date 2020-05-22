#pragma once
#include "frame.h"
#include "pixel_selector.h"
#include <opencv2/core.hpp>
#include <unordered_map>
namespace mpl {

struct Candidate {
    Eigen::Vector2i uv;
    float d_inv;
    float d_variance;
    bool is_converge = false;
    bool is_H_constrained = false;
};

class CandidateManager {
   public:
    CandidateManager() = default;
    void update_depth_per_frame(const Frame::ptr frame, bool is_KF);
    /**
     * @brief select candidate in frame, and initial depth with synetic_depth_im
     *
     * @param frame
     * @param synetic_depth_im
     */
    void select_candidate(const Frame::ptr frame,
                          const cv::Mat synetic_depth_im);

   private:
    // map frame ptr to it's candidate
    std::unordered_map<Frame::ptr, std::vector<Candidate>> candidate_map;
    // add candidate covariance info
    PixelSelector pixle_selector;
};
}  // namespace mpl
