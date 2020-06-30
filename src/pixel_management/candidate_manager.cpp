#include "candidate_manager.h"
#include "pattern.h"
#include <algorithm>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace mpl {
using namespace std;
using namespace Eigen;
using namespace Sophus;

inline bool is_in_image(CamData& cam, int u, int v) {
    return !(u > 1 && v > 1 && u < cam.width[0] - 2 && cam.height[0] - 2);
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
        can.u = p(0);
        can.v = p(1);
        can.is_depth_safe = !(mask.at<float>(can.v, can.u));
        float d = synetic_depth_im.at<ushort>(can.v, can.u);

        can.d_inv = (d == 0)
                        ? std::numeric_limits<float>::infinity()
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
    int dilation_size = 10;
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
    if (can.status == CandidateStatus::OOB) return;

    auto& cam = CamData::getInstance();

    Matrix3f KRKinv = cam.K[0] * T_new_old.rotationMatrix() * cam.K_inv[0];
    Vector3f Kt = cam.K[0] * T_new_old.translation();

    float max_search_range = (cam.width[0] + cam.height[0]) * 0.04f;
    Vector3f pR = KRKinv * Vector3f(can.u, can.v, 1);
    Vector3f p_far = pR + Kt * can.d_inv_min;

    float u_far = p_far.hnormalized()(0);
    float v_far = p_far.hnormalized()(1);

    if (is_in_image(cam, u_far, v_far)) {
        can.status = CandidateStatus::OOB;
        return;
    }

    float dist, u_near, v_near;
    Vector3f p_near;
    if (std::isfinite(can.d_inv_max)) {
        p_near = pR + Kt * can.d_inv_max;
        u_near = p_near.hnormalized()(0);
        v_near = p_near.hnormalized()(1);

        if (is_in_image(cam, u_far, v_far)) {
            can.status = CandidateStatus::OOB;
            return;
        }

        // ============== check their distance. everything below 2px is OK (->
        // skip). ===================
        dist = (p_far - p_near).head(2).norm();
        if (dist < 2.0f) {
            can.last_search_interval = 0;
            if (!isinf(can.d_inv) &&
                (std::abs(can.d_inv_line_search / can.d_inv - 1) < 5e-2)) {
                can.status = CandidateStatus::IS_MAP_POINT;
                return;
            }

            can.status = CandidateStatus::NOT_MAP_BUT_CONVERGE;
            return;
        }
    } else {
        dist = max_search_range;

        // project to arbitrary depth to get direction.
        p_near = pR + Kt * 0.01f;
        u_near = p_near.hnormalized()(0);
        v_near = p_near.hnormalized()(1);

        // direction.
        float d = 1.0f / (p_far - p_near).head(2).norm();

        u_near = u_far + dist * (u_near - u_far) * d;
        v_near = v_far + dist * (v_near - v_far) * d;

        // may still be out!
        if (is_in_image(cam, u_far, v_far)) {
            can.status = CandidateStatus::OOB;
            return;
        }
    }

    // set OOB if scale change too big.
    if (!(can.d_inv_min < 0 || (p_far[2] > 0.5 && p_far[2] < 1.5))) {
        can.status = CandidateStatus::OOB;
        return;
    }

    // ============== compute error-bounds on result in pixel. if the new
    // interval is not at least 1/2 of the old, SKIP ===================
    float dx = u_near - u_far;
    float dy = v_near - v_far;

    float a =
        (Vector2f(dx, dy).transpose() * can.structure_mat * Vector2f(dx, dy));
    float b =
        (Vector2f(dy, -dx).transpose() * can.structure_mat * Vector2f(dy, -dx));
    float errorInPixel = 0.2f + 0.2f * (a + b) / a;
    std::cout << "error in pixle " << errorInPixel << "      dist " << dist
              << "     last search interval " << can.last_search_interval
              << '\n';
    if (errorInPixel > 0.5 * dist && std::isfinite(can.d_inv_max)) {
        can.status = CandidateStatus::ILL_CONDITIONED;
        return;
    }

    if (errorInPixel > 10) {
        errorInPixel = 10;
        can.last_search_interval = 2 * errorInPixel;
        return;
    }

    // ============== do the discrete search ===================
    dx /= dist;
    dy /= dist;

    if (dist > max_search_range) {
        u_near = u_far + max_search_range * dx;
        v_near = v_far + max_search_range * dy;
        dist = max_search_range;
    }

    int numSteps = 1.9999f + dist;
    Matrix2f Rplane = KRKinv.topLeftCorner<2, 2>();

    float randShift = u_far * 1000 - floorf(u_far * 1000);
    float ptx = u_far - randShift * dx;
    float pty = v_far - randShift * dy;

    Vector2f rotatetPattern[8];
    for (int idx = 0; idx < 8; idx++)
        rotatetPattern[idx] =
            Rplane * Vector2f(pattern[idx][0], pattern[idx][1]);

    if (!std::isfinite(dx) || !std::isfinite(dy)) {
        can.status = CandidateStatus::OUTLIER;
        return;
    }

    float bestU = 0, bestV = 0, bestEnergy = 1e10;
    if (numSteps >= 100) numSteps = 99;

    auto image_pyramid = new_frame->getImagePyramid();
    for (int i = 0; i < numSteps; i++) {
        float energy = 0;
        for (int idx = 0; idx < 8; idx++) {
            float hitColor = image_pyramid->operator()(
                0, ptx + rotatetPattern[idx][0], pty + rotatetPattern[idx][1]);

            if (!std::isfinite(hitColor)) {
                energy += 1e5;
                continue;
            }
            float residual =
                hitColor - (float)(exp(aff_new_old.alpha()) * can.color[idx] +
                                   aff_new_old.beta());
            float hw = fabs(residual) < 15 ? 1 : 15 / fabs(residual);
            energy += hw * residual * residual * (2 - hw);
        }

        if (energy < bestEnergy) {
            bestU = ptx;
            bestV = pty;
            bestEnergy = energy;
        }

        ptx += dx;
        pty += dy;
    }

    // ============== do GN optimization ===================
    float uBak = bestU, vBak = bestV, gnstepsize = 1, stepBack = 0;
    bestEnergy = 1e5;
    int gnStepsGood = 0, gnStepsBad = 0;
    auto im_pyramid = *(new_frame->getImagePyramid());
    for (int it = 0; it < 3; it++) {
        float H = 1, b = 0, energy = 0;
        for (int idx = 0; idx < 8; idx++) {
            float u = bestU + rotatetPattern[idx][0];
            float v = bestV + rotatetPattern[idx][1];

            if (!std::isfinite(im_pyramid(0, u, v))) {
                energy += 1e5;
                continue;
            }
            float residual = im_pyramid(0, u, v) -
                             (exp(aff_new_old.alpha()) * can.color[idx] +
                              aff_new_old.beta());
            float dResdDist =
                dx * im_pyramid.dx(0, u, v) + dy * im_pyramid.dy(0, u, v);
            float hw = fabs(residual) < 15 ? 1 : 15 / fabs(residual);

            H += hw * dResdDist * dResdDist;
            b += hw * residual * dResdDist;
            energy += can.weight[idx] * can.weight[idx] * hw * residual *
                      residual * (2 - hw);
        }

        if (energy > bestEnergy) {
            gnStepsBad++;

            // do a smaller step from old point.
            stepBack *= 0.5;
            bestU = uBak + stepBack * dx;
            bestV = vBak + stepBack * dy;
        } else {
            gnStepsGood++;

            float step = -gnstepsize * b / H;
            if (step < -0.5)
                step = -0.5;
            else if (step > 0.5)
                step = 0.5;

            if (!std::isfinite(step)) step = 0;

            uBak = bestU;
            vBak = bestV;
            stepBack = step;

            bestU += step * dx;
            bestV += step * dy;
            bestEnergy = energy;
        }

        if (fabsf(stepBack) < 0.1) break;
    }

    if (!(bestEnergy < 20.f)) {
        can.status = (can.status == CandidateStatus::OUTLIER)
                         ? CandidateStatus::OOB
                         : CandidateStatus::OUTLIER;
        return;
    }

    // ============== set new interval ===================
    // dx here is cos , dy is sin
    float d_inv_min, d_inv_max;
    if (dx * dx > dy * dy) {
        d_inv_min = (pR[2] * (bestU - errorInPixel * dx) - pR[0]) /
                    (Kt[0] - Kt[2] * (bestU - errorInPixel * dx));
        d_inv_max = (pR[2] * (bestU + errorInPixel * dx) - pR[0]) /
                    (Kt[0] - Kt[2] * (bestU + errorInPixel * dx));
    } else {
        d_inv_min = (pR[2] * (bestV - errorInPixel * dy) - pR[1]) /
                    (Kt[1] - Kt[2] * (bestV - errorInPixel * dy));
        d_inv_max = (pR[2] * (bestV + errorInPixel * dy) - pR[1]) /
                    (Kt[1] - Kt[2] * (bestV + errorInPixel * dy));
    }
    if (d_inv_min > d_inv_max) std::swap<float>(d_inv_max, d_inv_min);

    if (!std::isfinite(d_inv_min) || !std::isfinite(d_inv_max) ||
        (d_inv_max < 0)) {
        can.status = CandidateStatus::OUTLIER;
        return;
    }

    float d_inv_line_search_old = can.d_inv_line_search;

    can.update(d_inv_min, d_inv_max);

    if (!std::isinf(can.d_inv)) {
        float diff = std::abs(can.d_inv / can.d_inv_line_search - 1);
        if (diff < 1e-2) {
            can.status = CandidateStatus::IS_MAP_POINT;
            can.last_search_interval = errorInPixel * 2;
            return;
        }
    }

    if (!std::isinf(d_inv_line_search_old)) {
        float diff =
            std::abs(can.d_inv_line_search / d_inv_line_search_old - 1);
        std::cout << "diff" << diff << '\n';

        if (diff < 1e-2) {
            can.status = CandidateStatus::NOT_MAP_BUT_CONVERGE;
            can.last_search_interval = errorInPixel * 2;
            return;
        } else if (diff > 2e-1) {
            can.status = CandidateStatus::OUTLIER;
            return;
        }
    }
    can.status = CandidateStatus::ILL_CONDITIONED;
    can.last_search_interval = errorInPixel * 2;

    return;
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
        float w = -std::sqrt(std::pow(can.color[idx] - can.color[0], 2) / 800);
        can.weight[idx] = exp(w);
        sum += can.weight[idx];
    }

    for (int i = 0; i < 8; ++i) {
        can.weight[i] /= sum;
    }
}

PointCloudPyramid::ptr CandidateManager::get_point_cloud_pyramid() {
    PointCloudPyramid::ptr pcp(new PointCloudPyramid);
    const int lvls = pcp->lvls();

    if (candidate_map.size() == 1) {
        int cnt = 0;
        for (auto& can : candidate_map[newst_KF]) {
            Eigen::Vector3f position = newst_KF->unproject(
                Eigen::Vector2i(can.u, can.v), 1.0f / can.d_inv);

            ++cnt;
            for (int lvl = 0; lvl < lvls; ++lvl) {
                if ((lvl != 0 && cnt % lvl == 0) || lvl == 0) {
                    (*pcp)[lvl].emplace_back(position, can.color[0]);
                }
            }
        }
    } else {
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
                    Eigen::Vector2i(can.u, can.v), can.d_inv_line_search);

                Eigen::Vector3f P_newst_KF = T_newst_host * P_host;
                Eigen::Vector2f p_newst_KF = newst_KF->project(P_newst_KF);

                if (!is_in_image(cam, p_newst_KF(0), p_newst_KF(1))) {
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
