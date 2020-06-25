#include "candidate_manager.h"
#include <algorithm>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace mpl {

void CandidateManager::update_depth_per_frame(const Frame::ptr frame,
                                              bool is_KF) {
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
        can.d_inv = 1.0f / synetic_depth_im.at<ushort>(can.v, can.u);
        candidate_vec.push_back(can);
    }

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

}  // namespace mpl
