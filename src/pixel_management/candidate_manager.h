#pragma once
#include "frame.h"
#include "pixel_selector.h"
#include <opencv2/core.hpp>
#include <unordered_map>
namespace mpl {

struct Candidate {
    int u;
    int v;
    float d_inv;
    bool is_depth_safe =
        false;  // has a small depth gradient on synetic depth image
    bool is_converge = false;            // depth almost not changed
    bool is_global_constrained = false;  // converge in 1st depth update
    bool is_active = false;
    float X, Y, Z;  // only for global point
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

    std::vector<Candidate> get_candidate(const Frame::ptr frame);

   private:
    cv::Mat generate_depth_safe_mask(const cv::Mat synetic_depth_im) const;

    // map frame ptr to it's candidate
    std::unordered_map<Frame::ptr, std::vector<Candidate>> candidate_map;
    // add candidate covariance info
    PixelSelector pixle_selector;
};
}  // namespace mpl
