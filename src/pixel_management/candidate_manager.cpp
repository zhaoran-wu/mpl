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

void CandidateManager::update_depth_per_frame(const Frame::ptr new_frame) {
    for (auto it = candidate_map.begin(); it != candidate_map.end(); ++it) {
        Frame::ptr old_frame = it->first;
        if (old_frame == new_frame) continue;

        SE3f T_new_old = get_src_to_dst_transform(old_frame, new_frame);
        AffineLight aff_new_old = get_src_to_dst_aff_light(old_frame, new_frame);

        for (auto& can : it->second) {
            update_depth_on_old_frame(can, T_new_old, aff_new_old, new_frame);
        }
    }
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

        can.is_depth_safe = !(mask.at<float>(can.v, can.u));

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
    if (!isfinite(can.d_inv_synetic_im)) return false;

    if (target_frame == nullptr) return true;

    const Vector2f& point_on_lastKF =
        target_frame->project(T_old_new * host_frame->unproject(Vector2i(can.u, can.v), can.d_inv_synetic_im));

    if (!target_frame->is_in_image(point_on_lastKF(0), point_on_lastKF(1))) return false;
    // valid

    float energy = 0;
    for (int i = 0; i < 8; ++i) {
        float u = point_on_lastKF(0) + rotated_pattern[i][0];
        float v = point_on_lastKF(1) + rotated_pattern[i][1];

        energy += can.weight[i] * abs(calc_light_diff(can.color[i], target_frame->at(u, v), aff_old_new));
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

void CandidateManager::update_depth_on_old_frame(Candidate& can, const Sophus::SE3f& T_new_old,
                                                 const AffineLight& aff_new_old, Frame::ptr new_frame) {
    if (can.status == CandidateStatus::OOB || can.status == CandidateStatus::BAD) return;

    auto& cam = CamData::getInstance();

    Matrix3f KRKinv = cam.K[0] * T_new_old.rotationMatrix() * cam.K_inv[0];
    Vector3f Kt = cam.K[0] * T_new_old.translation();

    Vector3f pR = KRKinv * Vector3f(can.u + 0.5f, can.v + 0.5f, 1);

    // get search area
    auto p_near_p_far = get_search_range(can, pR, Kt);
    const auto& p_near = p_near_p_far.first;
    const auto& p_far = p_near_p_far.second;

    if (!is_in_img(cam, p_near) && !is_in_img(cam, p_far)) {
        can.status = (can.status == CandidateStatus::NOT_INITIALIZED) ? CandidateStatus::BAD : CandidateStatus::OOB;
        return;
    }

    // calc corrected pattern
    Matrix2f Rplane = KRKinv.topLeftCorner<2, 2>();
    vector<Vector2f> rotatetPattern = get_rotatet_pattern(Rplane);

    // line search
    float u_best, v_best;
    float energy_best = line_search(u_best, v_best, can, new_frame, p_near, p_far, rotatetPattern, aff_new_old);

    if (!(energy_best < 50.f)) {
        if (can.status == CandidateStatus::NOT_INITIALIZED) {
            can.status = CandidateStatus::BAD;
        } else {
            can.status = (can.status == CandidateStatus::OUTLIER) ? CandidateStatus::OOB : CandidateStatus::OUTLIER;
        }
        return;
    }

    // optimization
    Vector2f vec_far_to_near = p_near - p_far;
    Vector2f direction = vec_far_to_near.normalized();
    float energy_best_optimize = optimize(u_best, v_best, can, direction, new_frame, rotatetPattern, aff_new_old);
    /*     std::cout << can.u << "   " << can.v << "    best u : " << u_best
                  << "  v_best : " << v_best << "   energy before :" <<
       energy_best
                  << "    energy after : " << energy_best_optimize << '\n' */
    ;

    // update
    update(u_best, v_best, can, direction, pR, Kt);
}

void CandidateManager::update(const float u_best, const float v_best, Candidate& can, const Eigen::Vector2f& direction,
                              const Eigen::Vector3f& pR, const Eigen::Vector3f& Kt) {
    float dx = direction(0);
    float dy = direction(1);

    // assert(abs(dx * dx + dy * dy - 1.f) < 1e-3);
    float a = (Vector2f(dx, dy).transpose() * can.structure_mat * Vector2f(dx, dy));
    float b = (Vector2f(dy, -dx).transpose() * can.structure_mat * Vector2f(dy, -dx));
    float errorInPixel = 0.2f + 0.2f * (a + b) / a;

    float d_inv_min_obs, d_inv_max_obs;
    if (dx * dx > dy * dy) {
        d_inv_min_obs = (pR[2] * (u_best - errorInPixel * dx) - pR[0]) / (Kt[0] - Kt[2] * (u_best - errorInPixel * dx));
        d_inv_max_obs = (pR[2] * (u_best + errorInPixel * dx) - pR[0]) / (Kt[0] - Kt[2] * (u_best + errorInPixel * dx));
    } else {
        d_inv_min_obs = (pR[2] * (v_best - errorInPixel * dy) - pR[1]) / (Kt[1] - Kt[2] * (v_best - errorInPixel * dy));
        d_inv_max_obs = (pR[2] * (v_best + errorInPixel * dy) - pR[1]) / (Kt[1] - Kt[2] * (v_best + errorInPixel * dy));
    }
    if (d_inv_min_obs > d_inv_max_obs) std::swap<float>(d_inv_max_obs, d_inv_min_obs);
    /*     std::cout << "Error :   " << errorInPixel << "  u:   " << can.u
                  << "    v:   " << can.v << "   min : " << d_inv_min_obs
                  << "   max : " << d_inv_max_obs << '\n'; */
    float d_inv_obs = (d_inv_max_obs + d_inv_min_obs) / 2;
    if ((d_inv_obs < 0)) {
        can.status = (can.status == CandidateStatus::NOT_INITIALIZED) ? CandidateStatus::BAD : CandidateStatus::OUTLIER;
        return;
    }
    float d_inv_old = can.d_inv;

    can.update(d_inv_obs, std::pow(d_inv_max_obs - d_inv_min_obs, 2));

    if (!std::isinf(can.d_inv_synetic_im)) {
        float diff = std::abs(can.d_inv_synetic_im / can.d_inv - 1);
        if (diff < 1e-2) {
            can.status = CandidateStatus::IS_MAP_POINT;
            return;
        }
    }

    if (d_inv_old != 0) {
        float diff = std::abs(can.d_inv / d_inv_old - 1);
        if (diff < 1e-2) {
            can.status = CandidateStatus::NOT_MAP_BUT_CONVERGE;
            return;
        }
    }
    can.status = CandidateStatus::ILL_CONDITIONED;
    return;
}

float CandidateManager::optimize(float& u_best, float& v_best, const Candidate& can, const Eigen::Vector2f& direction,
                                 const Frame::ptr frame, const vector<Eigen::Vector2f>& rotatetPattern,
                                 const AffineLight& aff) const {
    float u_old = u_best, v_old = v_best, gnstepsize = 1, stepBack = 0;
    int gnStepsGood = 0, gnStepsBad = 0;

    float energy_best = 1e5;

    for (int it = 0; it < 3; it++) {
        float H = 1, b = 0, energy = 0;
        for (int idx = 0; idx < 8; idx++) {
            float u = u_best + rotatetPattern[idx][0];
            float v = v_best + rotatetPattern[idx][1];

            if (!std::isfinite(frame->at(u, v))) {
                energy += 1e5;
                continue;
            }
            float residual = calc_light_diff(can.synetic_color[idx], (float)frame->at(u, v), aff);
            float dResdDist = direction(0) * frame->dx(u, v) + direction(1) * frame->dy(u, v);

            H += dResdDist * dResdDist;
            b += residual * dResdDist;
            energy += can.weight[idx] * can.weight[idx] * residual * residual;
        }

        if (energy > energy_best) {
            gnStepsBad++;

            // do a smaller step from old point.
            stepBack *= 0.5;
            u_best = u_old + stepBack * direction(0);
            v_best = v_old + stepBack * direction(1);
        } else {
            gnStepsGood++;

            float step = -gnstepsize * b / H;
            if (step < -0.5)
                step = -0.5;
            else if (step > 0.5)
                step = 0.5;

            if (!std::isfinite(step)) step = 0;

            u_old = u_best;
            v_old = v_best;
            stepBack = step;

            u_best += step * direction(0);
            v_best += step * direction(1);
            energy_best = energy;
        }

        if (fabsf(stepBack) < 0.1) break;
    }
    return energy_best;
}

std::pair<Eigen::Vector2f, Eigen::Vector2f> CandidateManager::get_search_range(const Candidate& can,
                                                                               const Eigen::Vector3f& pR,
                                                                               const Eigen::Vector3f& Kt) const {
    pair<Vector2f, Vector2f> p_near_p_far;

    if (can.status == CandidateStatus::NOT_INITIALIZED) {
        if (!isfinite(can.d_inv_synetic_im)) {
            p_near_p_far.second = pR.hnormalized();
            // p_near
            Vector3f p_near = pR + Kt * 0.01f;
            Vector2f direction = (p_near.hnormalized() - p_near_p_far.second).normalized();
            p_near_p_far.first = p_near_p_far.second + 40 * direction;
        } else {
            p_near_p_far.second = (pR + Kt * (0.3f * can.d_inv_synetic_im)).hnormalized();
            p_near_p_far.first = (pR + Kt * (1.7f * can.d_inv_synetic_im)).hnormalized();

            /*             std::cout << " range : "
                                  << (p_near_p_far.first -
               p_near_p_far.second).norm()
                                  << '\n' */
            ;
        }
    } else {
        // std::cout << " d_inv " << can.d_inv << "  var : " << can.var << '\n';
        p_near_p_far.second = (pR + Kt * can.get_d_inv_min()).hnormalized();
        p_near_p_far.first = (pR + Kt * can.get_d_inv_max()).hnormalized();
    }
    return p_near_p_far;
}

float CandidateManager::line_search(float& u_best, float& v_best, const Candidate& can, const Frame::ptr frame,
                                    const Eigen::Vector2f& p_near, const Eigen::Vector2f& p_far,
                                    const vector<Vector2f>& rotatetPattern, const AffineLight& aff) const {
    Vector2f vec_far_to_near = p_near - p_far;
    Vector2f direction = vec_far_to_near.normalized();

    float max_range_in_pixel = std::max(std::min(vec_far_to_near.norm(), 80.0f), 2.0f);
    float ptx = p_far(0), pty = p_far(1);

    float best_energy = std::numeric_limits<float>::max();
    CamData& cam = CamData::getInstance();
    for (int i = 0; i < max_range_in_pixel; i++) {
        float energy = 0;
        if (!is_in_img(cam, Eigen::Vector2f(ptx, pty))) continue;
        for (int idx = 0; idx < 8; idx++) {
            float hitColor = frame->at(ptx + rotatetPattern[idx][0], pty + rotatetPattern[idx][1]);

            if (!std::isfinite(hitColor)) {
                energy += 1e5;
                continue;
            }
            float residual = calc_light_diff(can.synetic_color[idx], hitColor, aff);

            energy += can.weight[idx] * can.weight[idx] * residual * residual;
        }

        if (energy < best_energy) {
            u_best = ptx;
            v_best = pty;
            best_energy = energy;
        }

        ptx += direction(0);
        pty += direction(1);
    }
    return best_energy;
}

void CandidateManager::calc_structure_mat(Frame::ptr host_frame, Candidate& can, cv::Mat weight_mask) {
    can.structure_mat.setZero();
    float sum = 0;
    for (int idx = 0; idx < 8; idx++) {
        int u = pattern[idx][0] + can.u;
        int v = pattern[idx][1] + can.v;

        Vector2f dxdy;
        dxdy(0) = host_frame->dx(u, v);
        dxdy(1) = host_frame->dy(u, v);

        can.color[idx] = host_frame->at(u, v);
        can.synetic_color[idx] = (float)host_frame->at_synetic(u, v);
        can.structure_mat += dxdy * dxdy.transpose();
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
            if (can.is_active == true || can.status == CandidateStatus::BAD) continue;

            //! decide now only with distmap
            // bool can_be_activated = (can.status != CandidateStatus::BAD &&
            //                         can.status != CandidateStatus::OOB);

            // if (!can_be_activated) continue;

            // check dist on dist map
            float dist = dist_map.dist(can.projection_on_newst_KF(0), can.projection_on_newst_KF(1));

            if (dist > static_cast<float>(min_square_dist)) {
                can.is_active = true;  // activate candidate
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
                if (can.age > 7 || can.status == CandidateStatus::BAD || !can.is_active) continue;
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
                if (dist_map.dist(can.u, can.v) < 12) continue;
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

    LOG(INFO) << "curr point cloud size :" << pcp->operator[](0).size();
    last_pcp = pcp;
    return pcp;
}
}  // namespace mpl
