#include "candidate_manager.h"
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

void CandidateManager::select_candidate(const Frame::ptr frame, const cv::Mat synetic_depth_im) {
    // frame is key frame --> set frame block data for optimization
    frame->get_frame_block()->setPose(frame->get_pose<Sophus::SE3f>().inverse().cast<double>());
    frame->get_frame_block()->setAffineLight(frame->get_aff_light());

    // remove oldst frame
    if (candidate_map.size() > 7) {
        candidate_map.erase(key_frames.front());
        key_frames.erase(key_frames.begin());
    }

    std::vector<Eigen::Vector3i> pixle_selected;
    pixle_selector.select(frame->getImagePyramid(), pixle_selected);

    cv::Mat mask = generate_depth_safe_mask(synetic_depth_im);
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

    for (const auto& p : pixle_selected) {
        Candidate can;
        can.host_frame = frame;
        if (p(2) != 0 || !is_in_img(CamData::getInstance(), p.head(2).cast<float>()))
            continue;  // choose a safe point only
        can.u = p(0);
        can.v = p(1);

        can.is_depth_safe = !(mask.at<float>(can.v, can.u));

        float d = synetic_depth_im.at<ushort>(can.v, can.u);

        can.d_inv_synetic_im = 1000.f / d;
        calc_structure_mat(frame, can);

        if (!is_synetic_depth_valid(can, frame, newst_KF, T_old_new, aff_old_new, rotated_pattern)) {
            can.d_inv_synetic_im = 0.0f;
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
        target_frame->project(T_old_new * host_frame->unproject(Vector2i(can.u, can.v), 1.0f / can.d_inv_synetic_im));

    if (!target_frame->is_in_image(0, point_on_lastKF(0), point_on_lastKF(1))) return false;
    // valid

    float energy = 0;
    for (int i = 0; i < 8; ++i) {
        float u = point_on_lastKF(0) + rotated_pattern[i][0];
        float v = point_on_lastKF(1) + rotated_pattern[i][1];

        energy += can.weight[i] * abs(target_frame->getImagePyramid()->operator()(0, u, v) -
                                      exp(aff_old_new.alpha()) * can.color[i] - aff_old_new.beta());
    }
    static int cnt = 0;

    // std::cout << " curr energy : " << energy << " outlier num: " << cnt <<
    // '\n';

    if (energy >= 30.f) ++cnt;
    return (energy < 30.0f);
}

cv::Mat CandidateManager::generate_depth_safe_mask(const cv::Mat synetic_depth_im) const {
    // calc depth gradient magnitude
    cv::Mat dx, dy;  // 1st derivative in x,y
    cv::Sobel(synetic_depth_im, dx, CV_32F, 1, 0);
    cv::Sobel(synetic_depth_im, dy, CV_32F, 0, 1);

    cv::Mat mag(dx.size(), dx.type());
    cv::Mat angle(dx.size(), dx.type());
    cv::cartToPolar(dx, dy, mag, angle);

    /*     for (int r = 0; r < mag.rows; ++r) {
            for (int c = 0; c < mag.cols; ++c) {
                std::cout << " r :" << r << ",  c :" << c << "--->   "
                          << mag.at<float>(r, c) << '\n';
            }
        }

    cv::imshow("mag", mag);
    cv::waitKey(0); */

    cv::Mat mask, mask_dilated;

    // create binary mask according to magnitude
    /*     cv::Mat mag_clone = mag.clone();
        std::nth_element(mag_clone.begin<float>(),
                         mag_clone.begin<float>() + mag.rows * mag.cols * 0.6f,
                         mag_clone.end<float>());*/
    ushort threshold_value = 15000;  //*(mag_clone.begin<float>() + mag.rows * mag.cols * 0.6f);

    cv::threshold(mag, mask, threshold_value, 1, cv::THRESH_BINARY);

    // dilation
    int dilation_size = 10 - candidate_map.size() / 3.f;  // 10
    cv::Mat element =
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
                                  cv::Point(dilation_size, dilation_size));
    /// Apply the dilation operation
    cv::dilate(mask, mask_dilated, element);

    /*     cv::imshow("mask not delated", mask);
        cv::imshow("final mask", mask_dilated);
        cv::waitKey(0); */

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
    auto im_pyramid = *(frame->getImagePyramid());

    float u_old = u_best, v_old = v_best, gnstepsize = 1, stepBack = 0;
    int gnStepsGood = 0, gnStepsBad = 0;

    float energy_best = 1e5;

    for (int it = 0; it < 3; it++) {
        float H = 1, b = 0, energy = 0;
        for (int idx = 0; idx < 8; idx++) {
            float u = u_best + rotatetPattern[idx][0];
            float v = v_best + rotatetPattern[idx][1];

            if (!std::isfinite(im_pyramid(0, u, v))) {
                energy += 1e5;
                continue;
            }
            float residual = im_pyramid(0, u, v) - (exp(aff.alpha()) * can.color[idx] + aff.beta());
            float dResdDist = direction(0) * im_pyramid.dx(0, u, v) + direction(1) * im_pyramid.dy(0, u, v);

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

    auto image_pyramid = frame->getImagePyramid();

    float max_range_in_pixel = std::max(std::min(vec_far_to_near.norm(), 80.0f), 2.0f);
    float ptx = p_far(0), pty = p_far(1);

    float best_energy = std::numeric_limits<float>::max();
    CamData& cam = CamData::getInstance();
    for (int i = 0; i < max_range_in_pixel; i++) {
        float energy = 0;
        if (!is_in_img(cam, Eigen::Vector2f(ptx, pty))) continue;
        for (int idx = 0; idx < 8; idx++) {
            float hitColor = image_pyramid->operator()(0, ptx + rotatetPattern[idx][0], pty + rotatetPattern[idx][1]);

            if (!std::isfinite(hitColor)) {
                energy += 1e5;
                continue;
            }
            float residual = hitColor - (exp(aff.alpha()) * can.color[idx] + aff.beta());

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

void CandidateManager::calc_structure_mat(Frame::ptr host_frame, Candidate& can) {
    auto ip = host_frame->getImagePyramid();
    can.structure_mat.setZero();
    float sum = 0;
    for (int idx = 0; idx < 8; idx++) {
        int u = pattern[idx][0] + can.u;
        int v = pattern[idx][1] + can.v;

        Vector2f dxdy;
        dxdy(0) = ip->dx(0, u, v);
        dxdy(1) = ip->dy(0, u, v);

        can.color[idx] = (float)ip->operator()(0, u, v);
        can.structure_mat += dxdy * dxdy.transpose();
        float w = -std::sqrt(std::pow(can.color[idx] - can.color[0], 2)) / 7;
        can.weight[idx] = exp(w);
        sum += can.weight[idx];
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
    float ratio = (float)num_active / 1500;

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

    min_dist_to_active = std::min(max(6, min_dist_to_active), 12);

    int min_square_dist = pow(min_dist_to_active, 2);

    // choose new active points
    std::cout << " before add ,we have active points : " << dist_map.get_num_obstacles()
              << " min_dist :" << min_square_dist << '\n';

    for (auto it = candidate_map.begin(); it != candidate_map.end(); ++it) {
        if (it->first == newst_KF) continue;
        for (auto& can : it->second) {
            if (can.is_active == true || can.status == CandidateStatus::BAD) continue;

            //! decide now only with distmap
            // bool can_be_activated = (can.status != CandidateStatus::BAD &&
            //                         can.status != CandidateStatus::OOB);

            // if (!can_be_activated) continue;

            // check dist on dist map
            float dist = dist_map.dist(can.projection_on_newst_KF(0), can.projection_on_newst_KF(1));

            if (dist > static_cast<float>(min_square_dist)) {
                // check and remove outlier
                // remove outlier
                float color = newst_KF->getImagePyramid()->operator()(0, can.projection_on_newst_KF(0),
                                                                      can.projection_on_newst_KF(1));
                auto aff = get_src_to_dst_aff_light(it->first, newst_KF);
                float color_diff = std::abs(color - exp(aff.alpha()) * can.color[0] - aff.beta());
                // std::cout << "color diff " << color_diff << '\n';
                if (color_diff > 70) {
                    continue;
                }

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

    std::cout << "current active points num:   " << dist_map.get_num_obstacles() << '\n';

    cv::Mat vis = dist_map.get_distance_map_for_visualization(true);
    /*     cv::imshow("dist_map ", vis);
        cv::waitKey(0) */
    ;
}
PointCloudPyramid::ptr CandidateManager::get_point_cloud_pyramid() {
    PointCloudPyramid::ptr pcp(new PointCloudPyramid);  // todo : avoid allocaction every time
    // before get a new reference point cloud we need to update our active point
    // set
    activate_candidate();
    if (!is_initialized) {
        const int lvls = pcp->lvls();
        if (candidate_map.size() < 1e10) {
            int cnt = 0;
            for (auto& can : candidate_map[newst_KF]) {
                if (!can.is_depth_safe || can.d_inv_synetic_im < 1e-20) continue;
                Eigen::Vector3f position =
                    newst_KF->unproject(Eigen::Vector2i(can.u, can.v), 1.0f / can.d_inv_synetic_im);

                ++cnt;
                for (int lvl = 0; lvl < lvls; ++lvl) {
                    if ((lvl != 0 && cnt % lvl == 0) || lvl == 0) {
                        (*pcp)[lvl].emplace_back(position, can.color[0]);
                        ++can.age;
                    }
                }
            }
            if (dist_map.get_num_obstacles() > 1500) is_initialized = true;
        }
    } else {
        const int lvls = pcp->lvls();
        if (candidate_map.size() < 1e10) {
            int cnt = 0;
            for (auto it = candidate_map.begin(); it != candidate_map.end(); ++it) {
                if (it->first == newst_KF) continue;
                for (auto& can : it->second) {
                    if (can.d_inv_synetic_im < 1e-20 || can.age > 7 || !can.is_depth_safe ||
                        can.status == CandidateStatus::BAD)
                        continue;
                    if (can.is_active) {
                        Eigen::Vector3f position =
                            (newst_KF->get_pose<Isometry3f>().inverse() * it->first->get_pose<Isometry3f>()) *
                            it->first->unproject(Eigen::Vector2i(can.u, can.v), 1.0f / can.d_inv_synetic_im);

                        ++cnt;
                        for (int lvl = 0; lvl < lvls; ++lvl) {
                            if ((lvl != 0 && cnt % lvl == 0) || lvl == 0) {
                                float color = newst_KF->getImagePyramid()->operator()(0, can.projection_on_newst_KF(0),
                                                                                      can.projection_on_newst_KF(1));
                                //! use color on newst kf to estimate the aff
                                //! param;

                                (*pcp)[lvl].emplace_back(position, color);
                                ++can.age;
                            }
                        }
                    }
                }
            }
            // for all can in newst KF(not active yet)
            for (auto& can : candidate_map[newst_KF]) {
                if (can.d_inv_synetic_im < 1e-20) continue;  // not valid

                if (dist_map.dist(can.u, can.v) < 25) continue;
                dist_map.add(Eigen::Vector2f(can.u, can.v));

                Eigen::Vector3f position =
                    newst_KF->unproject(Eigen::Vector2i(can.u, can.v), 1.0f / can.d_inv_synetic_im);

                ++cnt;
                for (int lvl = 0; lvl < lvls; ++lvl) {
                    if ((lvl != 0 && cnt % lvl == 0) || lvl == 0) {
                        (*pcp)[lvl].emplace_back(position, can.color[0]);
                        ++can.age;
                    }
                }
            }
        }
    }

    /*     const int lvls = pcp->lvls();
        if (candidate_map.size() < 3) {
            int cnt = 0;
            for (auto& can : candidate_map[newst_KF]) {
                if (!can.is_depth_safe) continue;
                Eigen::Vector3f position = newst_KF->unproject(
                    Eigen::Vector2i(can.u, can.v), 1.0f /
       can.d_inv_synetic_im);

                ++cnt;
                for (int lvl = 0; lvl < lvls; ++lvl) {
                    if ((lvl != 0 && cnt % lvl == 0) || lvl == 0) {
                        (*pcp)[lvl].emplace_back(position, can.color[0]);
                    }
                }
            }
        } */
    /* else {
        auto& cam = CamData::getInstance();
        for (auto it = candidate_map.begin(); it != candidate_map.end(); ++it) {
            Frame::ptr host_frame = it->first;
            if (host_frame == newst_KF) continue;
            auto candidates = it->second;
            SE3f T_host = host_frame->get_pose<SE3f>();
            SE3f T_newst = newst_KF->get_pose<SE3f>();
            SE3f T_newst_host = T_newst.inverse() * T_host;
            for (auto& can : candidates) {
                // todo
                if (can.status == CandidateStatus::ILL_CONDITIONED ||
                    can.status == CandidateStatus::OUTLIER) {
                    continue;
                }

                Eigen::Vector3f P_host = host_frame->unproject(
                    Eigen::Vector2i(can.u, can.v), can.d_inv);

                Eigen::Vector3f P_newst_KF = T_newst_host * P_host;
                Eigen::Vector2f p_newst_KF = newst_KF->project(P_newst_KF);

                if (!is_in_img(cam, p_newst_KF)) {
                    continue;
                }

                int cnt = 0;
                for (int lvl = 0; lvl < lvls; ++lvl) {
                    if ((lvl != 0 && cnt % lvl == 0) || lvl == 0) {
                        (*pcp)[lvl].emplace_back(P_newst_KF, can.color[0]);
                    }
                }
            }
        }
    }
 */
    return pcp;
}
}  // namespace mpl
