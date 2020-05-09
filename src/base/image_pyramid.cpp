#include "image_pyramid.h"
#include "../util/tictoc.h"
#include <algorithm>
#include <glog/logging.h>
namespace mpl {
ImagePyramid::ImagePyramid(const uchar* const row_data, const int lvls) {
    LOG_ASSERT(row_data != nullptr);
    cam_data = &CamData::getInstance();

    const uchar* parent_image = row_data;
    for (int lvl = 0; lvl < lvls; ++lvl) {
        build_image(parent_image, lvl);
        parent_image = image_pyramid[lvl].get();
        build_derivative(lvl);
    }
}

// generate image pyramid:  parent ---(samping)---> child
void ImagePyramid::build_image(const uchar* parent_image, const int child_lvl) {
    int w_child = cam_data->width[child_lvl];
    int h_child = cam_data->height[child_lvl];

    uchar_ptr child_ptr(new u_char[w_child * h_child]);
    int parent_step = cam_data->width[child_lvl - 1];
    if (child_lvl == 0) {
        memmove(child_ptr.get(), parent_image, w_child * h_child);
    } else {
        for (int u = 0; u < w_child; ++u) {
            for (int v = 0; v < h_child; ++v) {
                child_ptr[idx(w_child, u, v)] =
                    halfSampling(u, v, parent_step, parent_image);
            }
        }
    }
    image_pyramid.push_back(child_ptr);
}

// generate dx,dy, mag2 pyramid: direct compute with image in same lvl
void ImagePyramid::build_derivative(const int lvl) {
    int w_lvl = cam_data->width[lvl];
    int h_lvl = cam_data->height[lvl];

    uchar_ptr im_lvl = image_pyramid[lvl];
    float_ptr dx_lvl(new float[w_lvl * h_lvl]);
    float_ptr dy_lvl(new float[w_lvl * h_lvl]);
    float_ptr mag2_lvl(new float[w_lvl * h_lvl]);
    for (int u = 1; u < w_lvl - 1; ++u) {
        for (int v = 1; v < h_lvl - 1; ++v) {
            int curr_idx = idx(w_lvl, u, v);

            float dx_tmp = 0.5f * (im_lvl[curr_idx + 1] - im_lvl[curr_idx - 1]);
            dx_lvl[curr_idx] = dx_tmp;

            float dy_tmp =
                0.5f * (im_lvl[curr_idx + w_lvl] - im_lvl[curr_idx - w_lvl]);
            dy_lvl[curr_idx] = dy_tmp;

            mag2_lvl[curr_idx] = dx_tmp * dx_tmp + dy_tmp * dy_tmp;
        }
    }
    dx_pyramid.push_back(std::move(dx_lvl));
    dy_pyramid.push_back(std::move(dy_lvl));
    mag2_pyramid.push_back(std::move(mag2_lvl));
}

}  // namespace mpl
