#include "candidate_manager.h"
#include "pattern.h"
#include <algorithm>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace mpl {
using namespace std;
using namespace Eigen;
using namespace Sophus;

inline bool is_in_img(CamData& cam, const Eigen::Vector2f& p) {
    return (p(0) > 2 && p(1) > 2 && p(0) < cam.width[0] - 2 &&
            p(1) < cam.height[0] - 2);
}

void CandidateManager::update_depth_per_frame(const Frame::ptr new_frame) {
    Frame::ptr lasfKF = new_frame->get_ref_frame();

    Sophus::SE3f T_new_lastKF = new_frame->get_T_curr_lastKF();
    Sophus::SE3f T_w_lastKF = lasfKF->get_pose<Sophus::SE3f>();
    Sophus::SE3f T_new_w = T_new_lastKF * T_w_lastKF.inverse();

    AffineLight aff_w_lastKF = lasfKF->get_aff_light();
    AffineLight aff_new_lastKF = new_frame->get_aff_curr_lastKF();
    AffineLight aff_w_new =
        AffineLight::calc_dst_global_aff(aff_w_lastKF, aff_new_lastKF);

    for (auto it = candidate_map.begin(); it != candidate_map.end(); ++it) {
        Frame::ptr old_frame = it->first;
        if (old_frame == new_frame) continue;

        SE3f T_w_old = old_frame->get_pose<SE3f>().inverse();
        SE3f T_new_old = T_new_w * T_w_old;

        AffineLight aff_new_old = AffineLight::calc_aff_map_src_to_dst(
            old_frame->get_aff_light(), aff_w_new);

        for (auto& can : it->second) {
            update_depth_on_old_frame(can, T_new_old, aff_new_old, new_frame);
        }
    }
}

void CandidateManager::select_candidate(const Frame::ptr frame,
                                        const cv::Mat synetic_depth_im) {
    std::vector<Eigen::Vector3i> pixle_selected;
    pixle_selector.select(frame->getImagePyramid(), pixle_selected);

    cv::Mat mask = generate_depth_safe_mask(synetic_depth_im);
    std::vector<Candidate> candidate_vec;

    Candidate can;
    for (const auto& p : pixle_selected) {
        if (!is_in_img(CamData::getInstance(), p.head(2).cast<float>()))
            continue;
        can.u = p(0);
        can.v = p(1);
        can.is_depth_safe = !(mask.at<float>(can.v, can.u));
        float d = synetic_depth_im.at<ushort>(can.v, can.u);

        can.d_inv_synetic_im =
            (d == 0) ? std::numeric_limits<float>::infinity()
                     : 1000.f / synetic_depth_im.at<ushort>(can.v, can.u);
        calc_structure_mat(frame, can);
        candidate_vec.push_back(can);
    }
    newst_KF = frame;
    candidate_map[frame] = candidate_vec;
}

cv::Mat CandidateManager::generate_depth_safe_mask(
    const cv::Mat synetic_depth_im) const {
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
    ushort threshold_value =
        9000;  //*(mag_clone.begin<float>() + mag.rows * mag.cols * 0.6f);

    cv::threshold(mag, mask, threshold_value, 1, cv::THRESH_BINARY);

    // dilation
    int dilation_size = 8;  // 10
    cv::Mat element = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
        cv::Point(dilation_size, dilation_size));
    /// Apply the dilation operation
    cv::dilate(mask, mask_dilated, element);

    /*     cv::imshow("mask not delated", mask);
        cv::imshow("final mask", mask_dilated);
        cv::waitKey(0); */

    return mask_dilated;
}

std::vector<Candidate> CandidateManager::get_candidate(const Frame::ptr frame) {
    return candidate_map[frame];
}

void CandidateManager::update_depth_on_old_frame(Candidate& can,
                                                 const Sophus::SE3f T_new_old,
                                                 const AffineLight aff_new_old,
                                                 Frame::ptr new_frame) {
    if (can.status == CandidateStatus::OOB ||
        can.status == CandidateStatus::BAD)
        return;

    auto& cam = CamData::getInstance();

    Matrix3f KRKinv = cam.K[0] * T_new_old.rotationMatrix() * cam.K_inv[0];
    Vector3f Kt = cam.K[0] * T_new_old.translation();

    Vector3f pR = KRKinv * Vector3f(can.u, can.v, 1);

    // get search area
    auto p_near_p_far = get_search_range(can, pR, Kt);
    const auto& p_near = p_near_p_far.first;
    const auto& p_far = p_near_p_far.second;

    if (!is_in_img(cam, p_near) && !is_in_img(cam, p_far)) {
        can.status = (can.status == CandidateStatus::NOT_INITIALIZED)
                         ? CandidateStatus::BAD
                         : CandidateStatus::OOB;
        return;
    }

    // calc corrected pattern
    Matrix2f Rplane = KRKinv.topLeftCorner<2, 2>();
    Vector2f rotatetPattern[8];
    for (int idx = 0; idx < 8; idx++) {
        rotatetPattern[idx] =
            Rplane * Vector2f(pattern[idx][0], pattern[idx][1]);
    }

    // line search
    float u_best, v_best;
    float energy_best = line_search(u_best, v_best, can, new_frame, p_near,
                                    p_far, rotatetPattern, aff_new_old);

    if (!(energy_best < 50.f)) {
        if (can.status == CandidateStatus::NOT_INITIALIZED) {
            can.status = CandidateStatus::BAD;
        } else {
            can.status = (can.status == CandidateStatus::OUTLIER)
                             ? CandidateStatus::OOB
                             : CandidateStatus::OUTLIER;
        }
        return;
    }

    // optimization
    Vector2f vec_far_to_near = p_near - p_far;
    Vector2f direction = vec_far_to_near.normalized();
    float energy_best_optimize = optimize(
        u_best, v_best, can, direction, new_frame, rotatetPattern, aff_new_old);
    /*     std::cout << can.u << "   " << can.v << "    best u : " << u_best
                  << "  v_best : " << v_best << "   energy before :" <<
       energy_best
                  << "    energy after : " << energy_best_optimize << '\n' */
    ;

    // update
    update(u_best, v_best, can, direction, pR, Kt);
}

void CandidateManager::update(const float u_best, const float v_best,
                              Candidate& can, const Eigen::Vector2f& direction,
                              const Eigen::Vector3f& pR,
                              const Eigen::Vector3f& Kt) {
    float dx = direction(0);
    float dy = direction(1);

    assert(abs(dx * dx + dy * dy - 1.f) < 1e-3);
    float a =
        (Vector2f(dx, dy).transpose() * can.structure_mat * Vector2f(dx, dy));
    float b =
        (Vector2f(dy, -dx).transpose() * can.structure_mat * Vector2f(dy, -dx));
    float errorInPixel = 0.2f + 0.2f * (a + b) / a;

    float d_inv_min_obs, d_inv_max_obs;
    if (dx * dx > dy * dy) {
        d_inv_min_obs = (pR[2] * (u_best - errorInPixel * dx) - pR[0]) /
                        (Kt[0] - Kt[2] * (u_best - errorInPixel * dx));
        d_inv_max_obs = (pR[2] * (u_best + errorInPixel * dx) - pR[0]) /
                        (Kt[0] - Kt[2] * (u_best + errorInPixel * dx));
    } else {
        d_inv_min_obs = (pR[2] * (v_best - errorInPixel * dy) - pR[1]) /
                        (Kt[1] - Kt[2] * (v_best - errorInPixel * dy));
        d_inv_max_obs = (pR[2] * (v_best + errorInPixel * dy) - pR[1]) /
                        (Kt[1] - Kt[2] * (v_best + errorInPixel * dy));
    }
    if (d_inv_min_obs > d_inv_max_obs)
        std::swap<float>(d_inv_max_obs, d_inv_min_obs);
    /*     std::cout << "Error :   " << errorInPixel << "  u:   " << can.u
                  << "    v:   " << can.v << "   min : " << d_inv_min_obs
                  << "   max : " << d_inv_max_obs << '\n'; */
    float d_inv_obs = (d_inv_max_obs + d_inv_min_obs) / 2;
    if ((d_inv_obs < 0)) {
        can.status = (can.status == CandidateStatus::NOT_INITIALIZED)
                         ? CandidateStatus::BAD
                         : CandidateStatus::OUTLIER;
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

float CandidateManager::optimize(float& u_best, float& v_best,
                                 const Candidate& can,
                                 const Eigen::Vector2f& direction,
                                 const Frame::ptr frame,
                                 const Eigen::Vector2f* rotatetPattern,
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
            float residual = im_pyramid(0, u, v) -
                             (exp(aff.alpha()) * can.color[idx] + aff.beta());
            float dResdDist = direction(0) * im_pyramid.dx(0, u, v) +
                              direction(1) * im_pyramid.dy(0, u, v);

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

std::pair<Eigen::Vector2f, Eigen::Vector2f> CandidateManager::get_search_range(
    Candidate& can, const Eigen::Vector3f& pR, const Eigen::Vector3f& Kt) {
    pair<Vector2f, Vector2f> p_near_p_far;

    if (can.status == CandidateStatus::NOT_INITIALIZED) {
        if (!isfinite(can.d_inv_synetic_im)) {
            p_near_p_far.second = pR.hnormalized();
            // p_near
            Vector3f p_near = pR + Kt * 0.01f;
            Vector2f direction =
                (p_near.hnormalized() - p_near_p_far.second).normalized();
            p_near_p_far.first = p_near_p_far.second + 40 * direction;
        } else {
            p_near_p_far.second =
                (pR + Kt * (0.3f * can.d_inv_synetic_im)).hnormalized();
            p_near_p_far.first =
                (pR + Kt * (1.7f * can.d_inv_synetic_im)).hnormalized();

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

float CandidateManager::line_search(
    float& u_best, float& v_best, const Candidate& can, const Frame::ptr frame,
    const Eigen::Vector2f& p_near, const Eigen::Vector2f& p_far,
    const Vector2f* rotatetPattern, const AffineLight& aff) const {
    Vector2f vec_far_to_near = p_near - p_far;
    Vector2f direction = vec_far_to_near.normalized();

    auto image_pyramid = frame->getImagePyramid();

    float max_range_in_pixel =
        std::max(std::min(vec_far_to_near.norm(), 80.0f), 2.0f);
    float ptx = p_far(0), pty = p_far(1);

    float best_energy = std::numeric_limits<float>::max();
    CamData& cam = CamData::getInstance();
    for (int i = 0; i < max_range_in_pixel; i++) {
        float energy = 0;
        if (!is_in_img(cam, Eigen::Vector2f(ptx, pty))) continue;
        for (int idx = 0; idx < 8; idx++) {
            float hitColor = image_pyramid->operator()(
                0, ptx + rotatetPattern[idx][0], pty + rotatetPattern[idx][1]);

            if (!std::isfinite(hitColor)) {
                energy += 1e5;
                continue;
            }
            float residual =
                hitColor - (exp(aff.alpha()) * can.color[idx] + aff.beta());

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

void CandidateManager::calc_structure_mat(Frame::ptr host_frame,
                                          Candidate& can) {
    auto ip = host_frame->getImagePyramid();
    can.structure_mat.setZero();
    float sum = 0;
    for (int idx = 0; idx < 8; idx++) {
        float u = pattern[idx][0] + can.u;
        float v = pattern[idx][1] + can.v;

        Vector2f dxdy;
        dxdy(0) = ip->dx(0, u, v);
        dxdy(1) = ip->dy(0, u, v);

        can.color[idx] = ip->operator()(0, u, v);
        can.structure_mat += dxdy * dxdy.transpose();
        float w = -std::sqrt(std::pow(can.color[idx] - can.color[0], 2)) / 7;
        can.weight[idx] = exp(w);
        sum += can.weight[idx];
    }

    for (int i = 0; i < 8; ++i) {
        can.weight[i] = 0.125;  //= sum;
    }
}

PointCloudPyramid::ptr CandidateManager::get_point_cloud_pyramid() {
    PointCloudPyramid::ptr pcp(new PointCloudPyramid);
    const int lvls = pcp->lvls();
    if (candidate_map.size() < 10) {
        int cnt = 0;
        for (auto& can : candidate_map[newst_KF]) {
            if (!can.is_depth_safe) continue;
            Eigen::Vector3f position = newst_KF->unproject(
                Eigen::Vector2i(can.u, can.v), 1.0f / can.d_inv_synetic_im);

            ++cnt;
            for (int lvl = 0; lvl < lvls; ++lvl) {
                if ((lvl != 0 && cnt % lvl == 0) || lvl == 0) {
                    (*pcp)[lvl].emplace_back(position, can.color[0]);
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
                    Eigen::Vector2i(can.u, can.v), 1.0f / can.d_inv_synetic_im);

                ++cnt;
                for (int lvl = 0; lvl < lvls; ++lvl) {
                    if ((lvl != 0 && cnt % lvl == 0) || lvl == 0) {
                        (*pcp)[lvl].emplace_back(position, can.color[0]);
                    }
                }
            }
        } */
    else {
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

    return pcp;
}
}  // namespace mpl
