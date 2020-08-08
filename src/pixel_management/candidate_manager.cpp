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

const std::vector<Frame::ptr>& CandidateManager::get_key_frames() const {
    return key_frames;
}

CandidateManager::CandidateManager(std::shared_ptr<Visualizer> vis_ptr_) : vis_ptr(vis_ptr_) {
}

Frame::ptr CandidateManager::get_last_removed_kf() const {
    return this->to_remove_kf;
}

void CandidateManager::select_candidate(const Frame::ptr frame, const cv::Mat synetic_depth_im,
                                        cv::Mat alignment_mask) {
    Config& config = Config::getInstance();
    // set data for visulization
    to_remove_kf = (key_frames.size() >= config.WINDOW_SIZE) ? key_frames.front() : nullptr;
    this->vis_ptr->set_new_sliding_window_data(*this, to_remove_kf);

    // remove oldst frame
    if (candidate_map.size() >= config.WINDOW_SIZE) {
        // remove all candidate/observation, that  observated/host by oldest kf
        for (auto it = candidate_map.begin(); it != candidate_map.end(); ++it) {
            if (it->first == to_remove_kf) continue;
            for (auto& can : it->second) {
                for (auto it2 = can->observations.begin(); it2 != can->observations.end();) {
                    if (it2->second->host_frame() == to_remove_kf.get() ||
                        it2->second->obs_frame() == to_remove_kf.get()) {
                        it2 = can->observations.erase(it2);
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

    std::vector<std::unique_ptr<Candidate>> candidate_vec;
    // initialize candidate
    for (const auto& p : pixel_selected) {
        std::unique_ptr<Candidate> can(new Candidate());
        can->u = p(0);
        can->v = p(1);
        can->host_frame = frame;

        float d = synetic_depth_im.at<ushort>(can->v, can->u);

        can->d_inv_synetic_im = 1000.f / d;
        calc_structure_mat(frame, *can, alignment_mask);

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

std::vector<std::unique_ptr<Candidate>>& CandidateManager::get_candidate(const Frame::ptr frame) {
    return candidate_map[frame];
}

std::unordered_map<Frame::ptr, std::vector<std::unique_ptr<Candidate>>>& CandidateManager::get_candidate_map() {
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

PointCloudPyramid::ptr CandidateManager::get_point_cloud_pyramid() {
    PointCloudPyramid::ptr pcp(new PointCloudPyramid);  // in synetic image coordinate
    // before get a new reference point cloud we need to update our active point
    // set
    tictoc::tic();
    const int lvls = pcp->lvls();
    int cnt = 0;

    for (auto& can : candidate_map[newst_KF]) {
        can->status = CandidateStatus::ACTIVE;
        ++cnt;
        for (int lvl = 0; lvl < lvls; ++lvl) {
            if ((lvl != 0 && cnt % (lvl + 1) == 0) || lvl == 0) {
                Eigen::Vector3f position = newst_KF->unproject(Eigen::Vector2i(can->u, can->v), can->d_inv_synetic_im);

                (*pcp)[lvl].emplace_back(
                    can.get(), position,
                    newst_KF->at_synetic<float>(((can->u + 0.5f) / std::pow(2.0f, lvl)) - 0.5f,
                                                ((can->v + 0.5f) / std::pow(2.0f, lvl)) - 0.5f, lvl),
                    can->alignment_weight);
            }
        }
        ++can->age;
    }

    LOG(INFO) << "build pcp total using time: " << tictoc::toc() / 1000.0f << " ms ,"
              << "curr point cloud size of lvl 0: " << pcp->operator[](0).size();
    last_pcp = pcp;
    return pcp;
}
}  // namespace mpl
