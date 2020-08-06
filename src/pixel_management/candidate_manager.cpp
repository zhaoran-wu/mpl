#include "candidate_manager.h"
#include "../util/tictoc.h"
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

Frame::ptr CandidateManager::get_last_removed_kf() const {
    return this->to_remove_kf;
}
void add_last_tracking_result(PointCloudPyramid::ptr pcp, Frame::ptr newst_KF, CandidateManager& cm) {
    auto& cam = CamData::getInstance();
    for (auto& voxel : pcp->operator[](0)) {
        if ((voxel.can->status != CandidateStatus::ACTIVE && voxel.can->status != CandidateStatus::OOB) ||
            voxel.can->host_frame == cm.get_last_removed_kf() || voxel.can->host_frame == nullptr ||
            voxel.aligenment_weight < 1e-10)
            continue;

        if (voxel.can->point_block == nullptr) {
            voxel.can->point_block = std::make_unique<PointParameterBlock>(voxel.can->d_inv_synetic_im);

            // here only for new added active candidate
            auto& kf_vec = cm.get_key_frames();
            for (auto it = kf_vec.begin(); *it != voxel.can->host_frame; ++it) {
                if (*it == cm.get_last_removed_kf()) continue;
                Eigen::Vector2f projection = unproject_trans_project(*(voxel.can), voxel.can->host_frame, *it);

                // todo check energy
                if (is_in_img(cam, projection)) {
                    std::unique_ptr<PhotometricResidual> obs =
                        std::make_unique<PhotometricResidual>(voxel.can, it->get());
                    voxel.can->observations[*it] = std::move(obs);
                }
            }
        }

        // add all active point the new observation on newst KF
        std::unique_ptr<PhotometricResidual> obs_newst_KF =
            std::make_unique<PhotometricResidual>(voxel.can, newst_KF.get());
        voxel.can->observations[newst_KF] = std::move(obs_newst_KF);
    }
}
void CandidateManager::select_candidate(const Frame::ptr frame, const cv::Mat synetic_depth_im,
                                        cv::Mat alignment_mask) {
    // remove oldst frame
    Config& config = Config::getInstance();
    if (candidate_map.size() >= config.WINDOW_SIZE) {
        to_remove_kf = key_frames.front();
        // remove all candidate/observation, that  observated/host by oldest kf
        for (auto it = candidate_map.begin(); it != candidate_map.end(); ++it) {
            if (it->first == to_remove_kf) continue;
            for (auto& can : it->second) {
                for (auto it2 = can.observations.begin(); it2 != can.observations.end();) {
                    if (it2->second->host_frame() == to_remove_kf.get() ||
                        it2->second->obs_frame() == to_remove_kf.get()) {
                        it2 = can.observations.erase(it2);
                    } else {
                        it2++;
                    }
                }
            }
        }

        // remove all candidate host by to_remove_kf

        candidate_map.erase(to_remove_kf);
        key_frames.erase(key_frames.begin());
    }

    safe_mask = generate_depth_safe_mask(synetic_depth_im);

    std::vector<Eigen::Vector3i> pixel_selected;
    pixel_selector.select(frame, pixel_selected, safe_mask, alignment_mask);

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

        candidate_vec.push_back(std::move(can));
    }

    // set newst_KF to new add key frame
    key_frames.push_back(frame);
    newst_KF = frame;
    candidate_map[frame] = std::move(candidate_vec);
}

cv::Mat CandidateManager::generate_depth_safe_mask(const cv::Mat synetic_depth_im) {
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
            float depth = synetic_depth_im.at<ushort>(r, c);
            mag.at<float>(r, c) = (depth > 20000) ? 0.0f : mag.at<float>(r, c) / depth;
        }
    }

    cv::Mat mask, mask_dilated;

    float threshold_value = 0.6;
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
        float w1 = -std::abs(can.synetic_color[idx] - can.synetic_color[0]) / 32.0f;
        can.weight[idx] = std::sqrt(exp(w1));
        sum += can.weight[idx];

        can.alignment_weight = std::sqrt(std::exp(-((int)weight_mask.at<uchar>(can.v, can.u) / 32.0f)));
        // std::cout << "diff : " << (int)weight_mask.at<uchar>(can.v, can.u) << "   weight :" <<
        // can.alignment_weight
        //          << '\n';
    }

    for (int i = 0; i < 8; ++i) {
        can.weight[i] /= sum;
    }
}

void CandidateManager::activate_candidate() {
    auto& cam = CamData::getInstance();
    // pre-compute the distance map of (already) active points

    dist_map.compute(candidate_map, this->key_frames, safe_mask);

    // set the min dist to active based on statistics
    Config& config = Config::getInstance();

    int num_active = dist_map.get_num_obstacles();
    float ratio = (float)num_active / config.PIXEL_SELECTION_NUM;

    if (ratio > 1.7f)
        min_dist_to_active += 3;
    else if (ratio > 1.4f)
        min_dist_to_active += 2;
    else if (ratio > 1.1f)
        min_dist_to_active += 1;
    else if (ratio < 0.4f)  // less point is more dangerous than more point
        min_dist_to_active -= 4;
    else if (ratio < 0.7f)
        min_dist_to_active -= 3;
    else if (ratio < 0.9f)
        min_dist_to_active -= 2;

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

    tictoc::tic();
    if (!is_initialized) {  // initialized means,we have enough activated candidate
        const int lvls = pcp->lvls();
        int cnt = 0;

        for (auto& can : candidate_map[newst_KF]) {
            can.status = CandidateStatus::ACTIVE;
            ++cnt;
            for (int lvl = 0; lvl < lvls; ++lvl) {
                if ((lvl != 0 && cnt % (lvl + 1) == 0) || lvl == 0) {
                    Eigen::Vector3f position = newst_KF->unproject(Eigen::Vector2i(can.u, can.v), can.d_inv_synetic_im);

                    (*pcp)[lvl].emplace_back(
                        &can, position,
                        newst_KF->at_synetic<float>(((can.u + 0.5f) / std::pow(2.0f, lvl)) - 0.5f,
                                                    ((can.v + 0.5f) / std::pow(2.0f, lvl)) - 0.5f, lvl),
                        can.alignment_weight);
                }
            }
            ++can.age;
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
        }
        // for all can in newst KF(not active yet)
        for (auto& can : candidate_map[newst_KF]) {
            if (dist_map.dist(can.u, can.v) < 20) continue;

            can.status = CandidateStatus::ACTIVE;
            dist_map.add(Eigen::Vector2f(can.u, can.v));

            ++cnt;
            for (int lvl = 0; lvl < lvls; ++lvl) {
                if ((lvl != 0 && cnt % (lvl + 1) == 0) || lvl == 0) {
                    Eigen::Vector3f position = newst_KF->unproject(Eigen::Vector2i(can.u, can.v), can.d_inv_synetic_im);
                    (*pcp)[lvl].emplace_back(
                        &can, position,
                        newst_KF->at_synetic<float>(((can.u + 0.5f) / std::pow(2.0f, lvl)) - 0.5f,
                                                    ((can.v + 0.5f) / std::pow(2.0f, lvl)) - 0.5f, lvl),
                        can.alignment_weight);
                }
            }
            ++can.age;
        }
    }

    LOG(INFO) << "build pcp total using time: " << tictoc::toc() / 1000.0f << " ms ,"
              << "curr point cloud size of lvl 0: " << pcp->operator[](0).size();
    last_pcp = pcp;
    return pcp;
}
}  // namespace mpl
