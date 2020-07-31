#include "candidate_manager.h"
#include "debug.h"
#include "pattern.h"
#include <algorithm>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace mpl {
using namespace std;
using namespace Eigen;
using namespace Sophus;

vector<Vector2f> get_rotatet_pattern(const Matrix2f& R) {
    vector<Vector2f> rotatetPattern;
    for (int idx = 0; idx < 8; idx++) {
        rotatetPattern.push_back(R * Vector2f(pattern[idx][0], pattern[idx][1]));
    }
    return rotatetPattern;
}

std::vector<Frame::ptr>& CandidateManager::get_key_frames() {
    return key_frames;
}

CandidateManager::CandidateManager(std::shared_ptr<Visualizer> vis_ptr_) : vis_ptr(vis_ptr_) {
}

void CandidateManager::select_candidate(const Frame::ptr frame, const cv::Mat synetic_depth_im,
                                        cv::Mat alignment_mask) {
    // frame is key frame --> set frame block data for optimization
    frame->get_frame_block()->setPose(frame->get_pose().cast<double>());
    frame->get_frame_block()->setAffineLight(frame->get_aff_light());

    // remove oldst frame
    if (candidate_map.size() > 7) {
        candidate_map.erase(key_frames.front());
        key_frames.erase(key_frames.begin());
    }

    cv::Mat mask = generate_depth_safe_mask(synetic_depth_im);

    std::vector<Eigen::Vector3i> pixel_selected;
    pixel_selector.select(frame->get_synetic_photometric_pyramid(), pixel_selected, mask);

    std::vector<Candidate> candidate_vec;
    // todo reserve candidate

    SE3f T_old_new;
    AffineLight aff_old_new;
    vector<Vector2f> rotated_pattern;
    auto& cam = CamData::getInstance();
    if (newst_KF != nullptr) {
        T_old_new = get_src_to_dst_transform(frame, newst_KF);
        aff_old_new = get_src_to_dst_aff_light(frame, newst_KF);

        rotated_pattern =
            get_rotatet_pattern((cam.K[0] * T_old_new.rotationMatrix() * cam.K_inv[0]).topLeftCorner<2, 2>());
    }

    // initialize candidate
    for (const auto& p : pixel_selected) {
        Candidate can;
        can.host_frame = frame;
        if (!is_in_img(CamData::getInstance(), p.head(2).cast<float>())) continue;  // choose a safe point only
        can.u = p(0);
        can.v = p(1);

        float d = synetic_depth_im.at<ushort>(can.v, can.u);

        can.d_inv_synetic_im = 1000.f / d;
        calc_structure_mat(frame, can, alignment_mask);

        if (!is_synetic_depth_valid(can, frame, newst_KF, T_old_new, aff_old_new, rotated_pattern)) {
            continue;
        }
        candidate_vec.push_back(std::move(can));
    }
    // set newst_KF to new add key frame
    key_frames.push_back(frame);
    newst_KF = frame;
    candidate_map[frame] = std::move(candidate_vec);
}

bool CandidateManager::is_synetic_depth_valid(const Candidate& can, const Frame::ptr host_frame,
                                              const Frame::ptr target_frame, const SE3f T_old_new,
                                              const AffineLight& aff_old_new,
                                              const vector<Vector2f> rotated_pattern) const {
    if (target_frame == nullptr) return true;

    // project on last keyframe to validate
    const Vector2f& point_on_lastKF =
        target_frame->project(T_old_new * host_frame->unproject(Vector2i(can.u, can.v), can.d_inv_synetic_im));

    if (!target_frame->is_in_image(point_on_lastKF(0), point_on_lastKF(1))) return true;

    // valid

    float energy = 0;
    for (int i = 0; i < 8; ++i) {
        float u = point_on_lastKF(0) + rotated_pattern[i][0];
        float v = point_on_lastKF(1) + rotated_pattern[i][1];

        energy +=
            can.weight[i] * abs(calc_light_diff(can.synetic_color[i], target_frame->at_synetic(u, v), aff_old_new));
    }
    static int cnt = 0;

    // std::cout << " curr energy : " << energy << " outlier num: " << cnt << '\n';

    if (energy >= 30.f) ++cnt;  //!
    return (energy < 30.0f);
}

cv::Mat CandidateManager::generate_depth_safe_mask(const cv::Mat synetic_depth_im) const {
    auto& config = Config::getInstance();
    // calc depth gradient magnitude
    cv::Mat dx, dy;  // 1st derivative in x,y
    cv::Sobel(synetic_depth_im, dx, CV_32F, 1, 0);
    cv::Sobel(synetic_depth_im, dy, CV_32F, 0, 1);

    cv::Mat mag(dx.size(), dx.type());
    cv::Mat angle(dx.size(), dx.type());
    cv::cartToPolar(dx, dy, mag, angle);

    for (int r = 0; r < mag.rows; ++r) {
        for (int c = 0; c < mag.cols; ++c) {
            mag.at<float>(r, c) /= synetic_depth_im.at<ushort>(r, c);
        }
    }

    cv::Mat mask, mask_dilated;

    float threshold_value = 0.38;
    cv::threshold(mag, mask, threshold_value, 255, cv::THRESH_BINARY);

    // dilation
    int dilation_size = 4;  // 10
    cv::Mat element =
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
                                  cv::Point(dilation_size, dilation_size));
    /// Apply the dilation operation
    cv::dilate(mask, mask_dilated, element);

    debug::execute_func_according_to_config(config.DEBUG_DEPTH_SAFE_MASK, config.debug_depth_safe_mask_mutex, [&]() {
        cv::Mat result;
        cv::vconcat(mag, mask, result);
        cv::vconcat(result, mask_dilated, result);
        cv::resize(result, result, cv::Size(int(result.cols * 0.7), int(result.rows * 0.7)));
        cv::imshow("depth safe mask", result);
        cv::waitKey(0);
    });

    return mask_dilated;
}

std::vector<Candidate>& CandidateManager::get_candidate(const Frame::ptr frame) {
    return candidate_map[frame];
}

std::unordered_map<Frame::ptr, std::vector<Candidate>>& CandidateManager::get_candidate_map() {
    return candidate_map;
}

void CandidateManager::calc_structure_mat(Frame::ptr host_frame, Candidate& can, cv::Mat weight_mask) {
    float sum = 0;
    for (int idx = 0; idx < 8; idx++) {
        int u = pattern[idx][0] + can.u;
        int v = pattern[idx][1] + can.v;

        can.synetic_color[idx] = (float)host_frame->at_synetic(u, v);
        float w1 = -std::abs(can.synetic_color[idx] - can.synetic_color[0]) / 16.0f;
        can.weight[idx] = exp(w1);
        sum += can.weight[idx];

        can.alignment_weight = std::sqrt(std::exp(-((int)weight_mask.at<uchar>(can.v, can.u) / 16.0f)));
        // std::cout << "diff : " << (int)weight_mask.at<uchar>(can.v, can.u) << "   weight :" << can.alignment_weight
        //          << '\n';
    }

    for (int i = 0; i < 8; ++i) {
        can.weight[i] /= sum;
    }
}

void CandidateManager::activate_candidate() {
    auto& cam = CamData::getInstance();
    // pre-compute the distance map of (already) active points
    dist_map.compute(candidate_map, newst_KF);

    // set the min dist to active based on statistics
    Config& config = Config::getInstance();

    int num_active = dist_map.get_num_obstacles();
    float ratio = (float)num_active / (0.7f * config.PIXEL_SELECTION_NUM);

    if (ratio > 1.7f)
        min_dist_to_active += 3;
    else if (ratio > 1.4f)
        min_dist_to_active += 2;
    else if (ratio > 1.1f)
        min_dist_to_active += 1;
    else if (ratio < 0.3f)
        min_dist_to_active -= 3;
    else if (ratio < 0.6f)
        min_dist_to_active -= 2;
    else if (ratio < 0.9f)
        min_dist_to_active -= 1;

    min_dist_to_active = std::min(max(4, min_dist_to_active), 12);

    int min_square_dist = pow(min_dist_to_active, 2);

    // choose new active points
    // std::cout << " before add ,we have active points : " << dist_map.get_num_obstacles()
    //          << " min_dist :" << min_square_dist << '\n';

    for (auto it = candidate_map.begin(); it != candidate_map.end(); ++it) {
        if (it->first == newst_KF) continue;
        for (auto& can : it->second) {
            // remove bad candidate
            if (can.status != CandidateStatus::NOT_ACTIVE) continue;

            //! decide now only with distmap
            // bool can_be_activated = (can.status != CandidateStatus::BAD &&
            //                         can.status != CandidateStatus::OOB);

            // if (!can_be_activated) continue;

            // check dist on dist map
            float dist = dist_map.dist(can.projection_on_newst_KF(0), can.projection_on_newst_KF(1));

            if (dist > static_cast<float>(min_square_dist)) {
                can.status = CandidateStatus::ACTIVE;
                dist_map.add(can.projection_on_newst_KF);
                can.point_block->setIDepth((double)can.d_inv_synetic_im);  //! optimize based on map depth

                // check and add obsevations
                std::unique_ptr<PhotometricResidual> obs_newst_KF =
                    std::make_unique<PhotometricResidual>(&can, newst_KF);
                can.observations[newst_KF] = std::move(obs_newst_KF);

                for (auto it2 = candidate_map.begin(); it2 != candidate_map.end(); ++it2) {
                    if (it2->first == newst_KF || it2->first == can.host_frame) continue;
                    Eigen::Vector2f projection = unproject_trans_project(can, can.host_frame, it2->first);
                    if (is_in_img(cam, projection)) {
                        std::unique_ptr<PhotometricResidual> obs =
                            std::make_unique<PhotometricResidual>(&can, it2->first);
                        can.observations[it2->first] = std::move(obs);
                    }
                }
            }
        }
    }
    // debug only run if required in config file or setting in the menu
    debug::execute_mem_according_to_config(config.DEBUG_DISTANCE_MAP, config.debug_distance_map_mutex,
                                           &DistanceMap::show_distance_map_for_visualization, &dist_map, true);
}
PointCloudPyramid::ptr CandidateManager::get_point_cloud_pyramid() {
    PointCloudPyramid::ptr pcp(new PointCloudPyramid);  // todo : avoid allocaction every time
    // before get a new reference point cloud we need to update our active point
    // set
    activate_candidate();

    if (!is_initialized) {  // initialized means,we have enough activated candidate
        const int lvls = pcp->lvls();
        int cnt = 0;
        for (auto& can : candidate_map[newst_KF]) {
            ++cnt;
            for (int lvl = 0; lvl < lvls; ++lvl) {
                if ((lvl != 0 && cnt % lvl == 0) || lvl == 0) {
                    Eigen::Vector3f position = newst_KF->unproject(Eigen::Vector2i(can.u, can.v), can.d_inv_synetic_im);

                    (*pcp)[lvl].emplace_back(
                        &can, position,
                        newst_KF->at_synetic<float>(((can.u + 0.5f) / std::pow(2.0f, lvl)) - 0.5f,
                                                    ((can.v + 0.5f) / std::pow(2.0f, lvl)) - 0.5f, lvl));
                }
            }
        }
        if (dist_map.get_num_obstacles() > 1500) is_initialized = true;
    } else {
        const int lvls = pcp->lvls();
        int cnt = 0;
        for (auto it = candidate_map.begin(); it != candidate_map.end(); ++it) {
            if (it->first == newst_KF) continue;

            for (auto& can : it->second) {
                if (can.age > 7 || can.status != CandidateStatus::ACTIVE) continue;
                ++cnt;
                for (int lvl = 0; lvl < lvls; ++lvl) {
                    Eigen::Vector3f position =
                        get_src_to_dst_transform(it->first, newst_KF) *
                        it->first->unproject(Eigen::Vector2i(can.u, can.v), can.d_inv_synetic_im);

                    if ((lvl != 0 && cnt % (lvl + 1) == 0) || lvl == 0) {
                        float color = it->first->at_synetic<float>(((can.u + 0.5f) / std::pow(2.0f, lvl)) - 0.5f,
                                                                   ((can.v + 0.5f) / std::pow(2.0f, lvl)) - 0.5f, lvl);

                        (*pcp)[lvl].emplace_back(&can, position, color, can.alignment_weight);
                    }
                }
                ++can.age;
            }
            // for all can in newst KF(not active yet)
            for (auto& can : candidate_map[newst_KF]) {
                if (dist_map.dist(can.u, can.v) < 20) continue;
                dist_map.add(Eigen::Vector2f(can.u, can.v));

                ++cnt;
                for (int lvl = 0; lvl < lvls; ++lvl) {
                    if ((lvl != 0 && cnt % lvl == 0) || lvl == 0) {
                        Eigen::Vector3f position =
                            newst_KF->unproject(Eigen::Vector2i(can.u, can.v), can.d_inv_synetic_im);
                        (*pcp)[lvl].emplace_back(
                            &can, position,
                            newst_KF->at_synetic<float>(((can.u + 0.5f) / std::pow(2.0f, lvl)) - 0.5f,
                                                        ((can.v + 0.5f) / std::pow(2.0f, lvl)) - 0.5f, lvl));
                    }
                }
            }
        }
    }

    LOG(INFO) << "curr point cloud size of lvl 0: " << pcp->operator[](0).size();
    last_pcp = pcp;
    return pcp;
}
}  // namespace mpl
