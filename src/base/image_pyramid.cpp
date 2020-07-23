#include "image_pyramid.h"
#include "../util/tictoc.h"
#include "debug.h"
#include <algorithm>
#include <glog/logging.h>
#include <opencv2/highgui.hpp>
namespace mpl {
ImagePyramid::ImagePyramid(const uchar* const row_data) {
    LOG_ASSERT(row_data != nullptr);
    cam_data = &CamData::getInstance();
    auto& config = Config::getInstance();

    const uchar* parent_image = row_data;
    for (int lvl = 0; lvl < config.PYRAMID_LVLS; ++lvl) {
        build_image(parent_image, lvl);
        parent_image = image_pyramid[lvl].get();
        build_derivative(lvl);
    }

    debug::execute_mem_according_to_config(config.DEBUG_IMAGE_PYRAMID, config.debug_image_pyramid_mutex,
                                           &ImagePyramid::draw_result, this);
}

void ImagePyramid::draw_result() const {
    const int interval = 15;
    cv::Size size_result_im(cam_data->width[0] + 2 * interval, cam_data->height[0] * 1.5 + 3 * interval);

    cv::Mat im_result = cv::Mat::zeros(size_result_im, CV_8UC1);
    cv::Mat dx_result = cv::Mat::zeros(size_result_im, CV_32FC1);
    cv::Mat dy_result = cv::Mat::zeros(size_result_im, CV_32FC1);
    cv::Mat mag2_result = cv::Mat::zeros(size_result_im, CV_32FC1);

    int l = interval, u = interval;

    for (int lvl = 0; lvl < image_pyramid.size(); ++lvl) {
        cv::Size size_im(cam_data->width[lvl], cam_data->height[lvl]);
        cv::Mat im(size_im, CV_8UC1);
        cv::Mat dx_im(size_im, CV_32FC1);
        cv::Mat dy_im(size_im, CV_32FC1);
        cv::Mat mag2_im(size_im, CV_32FC1);

        cv::Rect roi(l, u, cam_data->width[lvl], cam_data->height[lvl]);

        cv::rectangle(im_result, roi, cv::Scalar(255, 0, 0), 2);

        im.data = data(lvl).get();
        dx_im.data = (uchar*)dx(lvl).get();
        dy_im.data = (uchar*)dy(lvl).get();
        mag2_im.data = (uchar*)mag_squared(lvl).get();

        // copy child to father
        im.copyTo(im_result(roi));
        dx_im.copyTo(dx_result(roi));
        dy_im.copyTo(dy_result(roi));
        mag2_im.copyTo(mag2_result(roi));

        // change child img location
        if (lvl & 0x1) {
            l += cam_data->width[lvl] + interval;
        } else {
            u += cam_data->height[lvl] + interval;
        }
    }

    cv::imshow("im pyramid", im_result);
    cv::imshow("dx pyramid", dx_result);
    cv::imshow("dy pyramid", dy_result);
    cv::imshow("mag2 pyramid", mag2_result);
    cv::waitKey(1);
}

// generate image pyramid:  parent ---(samping)---> child
void ImagePyramid::build_image(const uchar* parent_image, const int child_lvl) {
    int w_child = cam_data->width[child_lvl];
    int h_child = cam_data->height[child_lvl];

    uchar_ptr child_ptr(new u_char[w_child * h_child]);
    int parent_step = cam_data->width[child_lvl - 1];
    if (child_lvl == 0) {
        std::memcpy(child_ptr.get(), parent_image, w_child * h_child);
    } else {
        for (int u = 0; u < w_child; ++u) {
            for (int v = 0; v < h_child; ++v) {
                child_ptr[idx(w_child, u, v)] = halfSampling(u, v, parent_step, parent_image);
            }
        }
    }
    image_pyramid.push_back(child_ptr);
}

// generate dx,dy, mag_squared pyramid: direct compute with image in same lvl
void ImagePyramid::build_derivative(const int lvl) {
    int w_lvl = cam_data->width[lvl];
    int h_lvl = cam_data->height[lvl];
    int num_pixel = w_lvl * h_lvl;
    uchar_ptr im_lvl = image_pyramid[lvl];
    float_ptr dx_lvl(new float[num_pixel]);
    float_ptr dy_lvl(new float[num_pixel]);
    float_ptr mag2_lvl(new float[num_pixel]);

    memset(dx_lvl.get(), 0, sizeof(float) * num_pixel);
    memset(dy_lvl.get(), 0, sizeof(float) * num_pixel);
    memset(mag2_lvl.get(), 0, sizeof(float) * num_pixel);

    for (int u = 1; u < w_lvl - 1; ++u) {
        for (int v = 1; v < h_lvl - 1; ++v) {
            int curr_idx = idx(w_lvl, u, v);

            float dx_tmp = 0.5f * (im_lvl[curr_idx + 1] - im_lvl[curr_idx - 1]);
            dx_lvl[curr_idx] = dx_tmp;

            float dy_tmp = 0.5f * (im_lvl[curr_idx + w_lvl] - im_lvl[curr_idx - w_lvl]);
            dy_lvl[curr_idx] = dy_tmp;

            mag2_lvl[curr_idx] = dx_tmp * dx_tmp + dy_tmp * dy_tmp;
        }
    }
    dx_pyramid.push_back(std::move(dx_lvl));
    dy_pyramid.push_back(std::move(dy_lvl));
    mag2_pyramid.push_back(std::move(mag2_lvl));
}

}  // namespace mpl
