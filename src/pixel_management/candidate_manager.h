#pragma once
#include "frame.h"
#include "pixel_selector.h"
#include "point_cloud_pyramid.h"
#include <opencv2/core.hpp>
#include <unordered_map>
namespace mpl {

/**
 * @brief the status of candidate in the last search
 *
 */
enum class CandidateStatus {
    NOT_INITIALIZED,  // do not have benn searched even once
    ILL_CONDITIONED,  // epi-polar line direction almost parallel with gradient
                      // in last search
    IS_MAP_POINT,     // d_inv is converge and close to d_inv after
                      // some search
    NOT_MAP_BUT_CONVERGE,  // d_inv is converge but not close to
                           // d_inv
    OUTLIER,  // minimal energy larger than threshold, if occur twice --> out of
              // boundary
    OOB,      // out of boundary --> to be marginalize
    BAD       // can not be used
};

struct Candidate {
    float color[8];   // pattern color
    float weight[8];  // pattern weight
    int u;
    int v;
    float d_inv_synetic_im;  // in m, initialized with synetic depth map

    // line search
    float d_inv = 0;  // var is max, set d_inv to random value;
    float var = std::numeric_limits<float>::max();
    void update(float d_inv_obs, float var_obs);
    float get_d_inv_min();
    float get_d_inv_max();

    Eigen::Matrix2f structure_mat;
    // bool is_oob = false;  // is oob or outlier
    bool is_depth_safe =
        false;  // has a small depth gradient on synetic depth image
    // bool is_converge = false;   // d_inv almost not changed
    // bool is_map_point = false;  //  d_inv almost not change in the first
    // update float delta_d; bool is_active = false;
    CandidateStatus status = CandidateStatus::NOT_INITIALIZED;
};

class CandidateManager {
   public:
    CandidateManager() = default;
    void update_depth_per_frame(const Frame::ptr frame);
    /**
     * @brief select candidate in key frame, and initial depth with
     * synetic_depth_im
     *
     * @param frame
     * @param synetic_depth_im
     */
    void select_candidate(const Frame::ptr frame,
                          const cv::Mat synetic_depth_im);

    PointCloudPyramid::ptr get_point_cloud_pyramid();

    std::vector<Candidate> get_candidate(const Frame::ptr frame);

   private:
    cv::Mat generate_depth_safe_mask(const cv::Mat synetic_depth_im) const;

    void update_depth_on_old_frame(Candidate& can,
                                   const Sophus::SE3f T_curr_old,
                                   const AffineLight aff_curr_old,
                                   Frame::ptr curr_frame);

    void calc_structure_mat(Frame::ptr host_frame, Candidate& can);

    // map frame ptr to it's candidate
    std::unordered_map<Frame::ptr, std::vector<Candidate>> candidate_map;
    Frame::ptr newst_KF;
    // add candidate covariance info
    PixelSelector pixle_selector;
};

inline void Candidate::update(float d_inv_obs, float var_obs) {
    d_inv = (d_inv * var_obs + d_inv_obs * var) / (var + var_obs);
    var = var * var_obs / (var + var_obs);
    std::cout << " var " << var << '\n';
}
inline float Candidate::get_d_inv_min() {
    float tmp = d_inv - sqrt(var);
    return (tmp < 0) ? 0 : tmp;
}

inline float Candidate::get_d_inv_max() {
    return d_inv + sqrt(var);
}
}  // namespace mpl
