#pragma once
#include "ceres/PhotometricResidual.h"
#include "ceres/PointParameterBlock.h"
#include "distance_map.h"
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
    float d_inv_synetic_im;  // in m, initialized with synetic depth map, = 0 if
                             // the synetic depth is not valid

    // line search
    float d_inv = 0;  // var is max, set d_inv to random value;
    float var = 1e10;

    void update(float d_inv_obs, float var_obs);
    float get_d_inv_min() const;
    float get_d_inv_max() const;
    const std::unique_ptr<PointParameterBlock>& get_point_block() const;
    void merge_optimization_result();
    Eigen::Matrix2f structure_mat;

    bool is_active = false;  // only active will join the optimization
    bool is_depth_safe =
        false;  // has a small depth gradient on synetic depth image
    // bool is_converge = false;   // d_inv almost not changed
    // bool is_map_point = false;  //  d_inv almost not change in the first
    // update float delta_d; bool is_active = false;
    CandidateStatus status = CandidateStatus::NOT_INITIALIZED;

    // use for activate
    Eigen::Vector2f projection_on_newst_KF = Eigen::Vector2f(0, 0);
    int age = 0;

    // host frame
    Frame::ptr host_frame;

    std::unordered_map<Frame::ptr, std::unique_ptr<PhotometricResidual>>
        observations;

    // ceres optimization param
    std::unique_ptr<PointParameterBlock> point_block;
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

    std::vector<Candidate>& get_candidate(const Frame::ptr frame);

    std::unordered_map<Frame::ptr, std::vector<Candidate>>& get_candidate_map();

    void activate_candidate();

    std::vector<Frame::ptr>& get_key_frames();

   private:
    bool is_synetic_depth_valid(
        const Candidate& can, const Frame::ptr host_frame,
        const Frame::ptr target_frame, const Sophus::SE3f T_old_new,
        const AffineLight& aff_old_new,
        const std::vector<Eigen::Vector2f> rotated_pattern) const;

    std::pair<Eigen::Vector2f, Eigen::Vector2f> get_search_range(
        const Candidate& can, const Eigen::Vector3f& pR,
        const Eigen::Vector3f& Kt) const;
    cv::Mat generate_depth_safe_mask(const cv::Mat synetic_depth_im) const;

    // line search part
    float line_search(float& u_best, float& v_best, const Candidate& can,
                      const Frame::ptr frame, const Eigen::Vector2f& p_near,
                      const Eigen::Vector2f& p_far,
                      const std::vector<Eigen::Vector2f>& rotatetPattern,
                      const AffineLight& aff) const;

    float optimize(float& u_best, float& v_best, const Candidate& can,
                   const Eigen::Vector2f& direction, const Frame::ptr frame,
                   const std::vector<Eigen::Vector2f>& rotatetPattern,
                   const AffineLight& aff) const;

    void update(const float u_best, const float v_best, Candidate& can,
                const Eigen::Vector2f& vec_far_to_near,
                const Eigen::Vector3f& pR, const Eigen::Vector3f& Kt);
    void update_depth_on_old_frame(Candidate& can,
                                   const Sophus::SE3f& T_new_old,
                                   const AffineLight& aff_new_old,
                                   Frame::ptr new_frame);

    void calc_structure_mat(Frame::ptr host_frame, Candidate& can);

    // map frame ptr to it's candidate
    std::unordered_map<Frame::ptr, std::vector<Candidate>> candidate_map;
    Frame::ptr newst_KF = nullptr;
    std::vector<Frame::ptr> key_frames;
    // add candidate covariance info
    PixelSelector pixle_selector;
    // distance map
    DistanceMap dist_map;
    int min_dist_to_active = 5;

    bool is_initialized = false;
};

inline void Candidate::update(float d_inv_obs, float var_obs) {
    d_inv = (d_inv * var_obs + d_inv_obs * var) / (var + var_obs);
    var = var * var_obs / (var + var_obs);

    /*     std::cout << " u : " << u << "  v : " << v << "    d_inv : " << d_inv
                  << "  var: " << var << '\n' */
    ;
}
inline float Candidate::get_d_inv_min() const {
    float tmp = d_inv - sqrt(var);
    return (tmp < 0) ? 0 : tmp;
}

inline float Candidate::get_d_inv_max() const {
    return d_inv + sqrt(var);
}

inline const std::unique_ptr<PointParameterBlock>& Candidate::get_point_block()
    const {
    return point_block;
}

template <typename T>
inline bool is_in_img(CamData& cam, const T& p) {
    return (p(0) > 2 && p(1) > 2 && p(0) < cam.width[0] - 2 &&
            p(1) < cam.height[0] - 2);
}

// project with inv depth on synetic image

inline Eigen::Vector2f unproject_trans_project(const Candidate& can,
                                               const Frame::ptr host_frame,
                                               const Frame::ptr target_frame) {
    Eigen::Vector3f P_host = host_frame->unproject(
        Eigen::Vector2i(can.u, can.v), 1.f / can.d_inv_synetic_im);
    Eigen::Isometry3f T_target_host =
        target_frame->get_pose<Eigen::Isometry3f>().inverse() *
        host_frame->get_pose<Eigen::Isometry3f>();
    return target_frame->project(T_target_host * P_host);
}

inline void Candidate::merge_optimization_result() {
    this->d_inv = point_block->getIDepth();
}
}  // namespace mpl
