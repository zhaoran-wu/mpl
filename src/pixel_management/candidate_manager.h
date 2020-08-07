#pragma once
#include "../visualizer/visualizer.h"
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
    NOT_ACTIVE,  // never been used to generate a tracking reference point cloud
    ACTIVE,      // used to generate a tracking referenc point cloud,(good track >= 0, bad track = 0 )
    OUTLIER,     // good track = 0 , bad track = 1;
    OOB,         // out of boundary: good track > 0, bad track >= 1;
                 // 1. good track > 0 but not in curr imag(bad track = 1)
                 // 2. occlusion : good track > 0, bad track > 1
};

struct Candidate {
    float synetic_color[8];  // pattern color of synetic image
    float weight[8];         // pattern weight of synetic image
    float alignment_weight = 1;

    int u;
    int v;
    std::mutex d_inv_mutex;
    float d_inv_synetic_im;  // in m, initialized with synetic depth map, = 0 if
                             // the synetic depth is not valid
    int good_track_cnt = 0;
    int bad_track_cnt = 0;

    const std::unique_ptr<PointParameterBlock>& get_point_block() const;
    void merge_optimization_result();
    CandidateStatus status = CandidateStatus::NOT_ACTIVE;

    // use for activate
    Eigen::Vector2f projection_on_newst_KF = Eigen::Vector2f(0, 0);
    int age = 0;

    // host frame
    Frame::ptr host_frame;

    std::unordered_map<Frame::ptr, std::unique_ptr<PhotometricResidual>> observations;

    // ceres optimization param
    std::unique_ptr<PointParameterBlock> point_block = nullptr;
};

class CandidateManager {
   public:
    CandidateManager(std::shared_ptr<Visualizer> vis_ptr);

    /**
     * @brief select candidate in key frame, and initial depth with
     * synetic_depth_im
     *
     * @param frame
     * @param synetic_depth_im
     */
    void select_candidate(const Frame::ptr frame, const cv::Mat synetic_depth_im, cv::Mat alignment_mask);

    PointCloudPyramid::ptr get_point_cloud_pyramid();

    std::vector<std::unique_ptr<Candidate>>& get_candidate(const Frame::ptr frame);

    std::unordered_map<Frame::ptr, std::vector<std::unique_ptr<Candidate>>>& get_candidate_map();

    void activate_candidate();

    const std::vector<Frame::ptr>& get_key_frames() const;
    Frame::ptr get_last_removed_kf() const;

   private:
    // valid : depth from model is nearly correct
    bool is_synetic_depth_valid(const Candidate& can, const Frame::ptr host_frame, const Frame::ptr target_frame,
                                const Sophus::SE3f T_old_new, const AffineLight& aff_old_new,
                                const std::vector<Eigen::Vector2f> rotated_pattern) const;

    // safe depth has a vuale 0 in the mask
    cv::Mat generate_depth_safe_mask(const cv::Mat synetic_depth_im);

    void calc_structure_mat(Frame::ptr host_frame, Candidate& can, cv::Mat weight_mask);

    // map frame ptr to it's candidate
    std::unordered_map<Frame::ptr, std::vector<std::unique_ptr<Candidate>>> candidate_map;

    Frame::ptr newst_KF = nullptr;
    std::vector<Frame::ptr> key_frames;
    // add candidate covariance info
    PixelSelector pixel_selector;
    // distance map
    DistanceMap dist_map;
    int min_dist_to_active = 5;

    bool is_initialized = false;  // initialized: has enough activate point, else directly use synetic depth image in
                                  // curr frame to tracking
    std::shared_ptr<Visualizer> vis_ptr;

    PointCloudPyramid::ptr last_pcp = nullptr;

    Frame::ptr to_remove_kf = nullptr;

    cv::Mat safe_mask;
};

inline const std::unique_ptr<PointParameterBlock>& Candidate::get_point_block() const {
    return point_block;
}

template <typename T>
inline bool is_in_img(CamData& cam, const T& p) {
    return (p(0) > 2 && p(1) > 2 && p(0) < cam.width[0] - 2 && p(1) < cam.height[0] - 2);
}

// project with inv depth on synetic image

inline Eigen::Vector2f unproject_trans_project(const Candidate* const can, const Frame::ptr host_frame,
                                               const Frame::ptr target_frame) {
    Eigen::Vector3f P_host = host_frame->unproject(Eigen::Vector2i(can->u, can->v), can->d_inv_synetic_im);
    Sophus::SE3f T_target_host = get_src_to_dst_transform(host_frame, target_frame);
    return target_frame->project(T_target_host * P_host);
}

inline void Candidate::merge_optimization_result() {
    // std::cout << "old_depth :" << this->d_inv_synetic_im;
    d_inv_mutex.lock();
    this->d_inv_synetic_im = (float)this->point_block->getIDepth();
    d_inv_mutex.unlock();

    // std::cout << "  new_depth :" << this->d_inv_synetic_im << '\n';
}
}  // namespace mpl
