#pragma once
#include "cam_data.h"
#include "config.h"
#include <memory>
#include <vector>
namespace mpl {
typedef unsigned char uchar;

/**
 * @brief accept 2D coordinate and step(e.g image width )to calc 1D idx
 *
 */
inline int idx(const int step, const int u, const int v);

/**
 * @brief ImagePyramid access a uchar* to build a image image_pyramid of
 * lvls
 *
 */
class ImagePyramid {
   public:
    typedef std::shared_ptr<ImagePyramid> ptr;
    typedef std::shared_ptr<uchar[]> uchar_ptr;
    typedef std::shared_ptr<float[]> float_ptr;

    ImagePyramid(const uchar* const row_data);

    // data :row intensity , dx,dy: derivative in x and y direction
    // get image at lvl
    uchar_ptr data(const int lvl);
    float_ptr dx(const int lvl);
    float_ptr dy(const int lvl);
    float_ptr mag2(const int lvl);
    // get pixel value at lvl
    uchar operator()(const int lvl, const int u, const int v) const;
    float dy(const int lvl, const int u, const int v) const;
    float dx(const int lvl, const int u, const int v) const;
    float mag2(const int lvl, const int u, const int v) const;

    // sub-pixel
    float operator()(const int lvl, const float u, const float v) const;
    float dx(const int lvl, const float u, const float v) const;
    float dy(const int lvl, const float u, const float v) const;
    // todo overload for eigen vec
    bool is_in_image(const int lvl, const int u, const int v) const;

    int lvls() const;

   private:
    void build_image(const uchar* parent_image, const int lvl);
    void build_derivative(const int lvl);
    std::vector<uchar_ptr> image_pyramid;
    std::vector<float_ptr> dx_pyramid;
    std::vector<float_ptr> dy_pyramid;
    std::vector<float_ptr> mag2_pyramid;  // dx^2 + dy^2
    CamData* cam_data;
};

//########################################################//
//#####################implementation#####################//

inline ImagePyramid::uchar_ptr ImagePyramid::data(const int lvl) {
    return this->image_pyramid[lvl];
}

inline ImagePyramid::float_ptr ImagePyramid::dx(const int lvl) {
    return this->dx_pyramid[lvl];
}

inline ImagePyramid::float_ptr ImagePyramid::dy(const int lvl) {
    return this->dy_pyramid[lvl];
}

inline ImagePyramid::float_ptr ImagePyramid::mag2(const int lvl) {
    return this->mag2_pyramid[lvl];
}

inline bool ImagePyramid::is_in_image(const int lvl, const int u,
                                      const int v) const {
    return (u > 0 && u < cam_data->width[lvl] && v > 0 &&
            v < cam_data->height[lvl]);
}

inline uchar ImagePyramid::operator()(const int lvl, const int u,
                                      const int v) const {
    return this->image_pyramid[lvl][v * cam_data->width[lvl] + u];
}

inline float ImagePyramid::dy(const int lvl, const int u, const int v) const {
    return this->dy_pyramid[lvl][idx(cam_data->width[lvl], u, v)];
}

inline float ImagePyramid::dx(const int lvl, const int u, const int v) const {
    return this->dx_pyramid[lvl][idx(cam_data->width[lvl], u, v)];
}

// u,v here is hit pixel position
inline float ImagePyramid::operator()(const int lvl, const float u,
                                      const float v) const {
    int u_max = ceil(u - 0.5f);
    int v_max = ceil(v - 0.5f);
    int u_min = u_max - 1;
    int v_min = v_max - 1;

    float x = std::abs(u - 0.5f - u_min);
    float y = std::abs(v - 0.5f - v_min);

    if (u_max == cam_data->width[lvl] || v_max == cam_data->height[lvl]) {
        return this->operator()(lvl, u_max, v_max);
    }
    return this->operator()(lvl, u_max, v_max) * x * y +
           this->operator()(lvl, u_min, v_max) * (1 - x) * y +
           this->operator()(lvl, u_max, v_min) * x * (1 - y) +
           this->operator()(lvl, u_min, v_min) * (1 - x) * (1 - y);
}

inline float ImagePyramid::dy(const int lvl, const float u,
                              const float v) const {
    int u_max = ceil(u - 0.5f);
    int v_max = ceil(v - 0.5f);
    int u_min = u_max - 1;
    int v_min = v_max - 1;

    float x = std::abs(u - 0.5f - u_min);
    float y = std::abs(v - 0.5f - v_min);

    if (u_max == cam_data->width[lvl] || v_max == cam_data->height[lvl]) {
        return this->dy(lvl, u_max, v_max);
    }
    return this->dy(lvl, u_max, v_max) * x * y +
           this->dy(lvl, u_min, v_max) * (1 - x) * y +
           this->dy(lvl, u_max, v_min) * x * (1 - y) +
           this->dy(lvl, u_min, v_min) * (1 - x) * (1 - y);
}

inline float ImagePyramid::dx(const int lvl, const float u,
                              const float v) const {
    int u_max = ceil(u - 0.5f);
    int v_max = ceil(v - 0.5f);
    int u_min = u_max - 1;
    int v_min = v_max - 1;

    float x = std::abs(u - 0.5f - u_min);
    float y = std::abs(v - 0.5f - v_min);

    if (u_max == cam_data->width[lvl] || v_max == cam_data->height[lvl]) {
        return this->dx(lvl, u_max, v_max);
    }
    return this->dx(lvl, u_max, v_max) * x * y +
           this->dx(lvl, u_min, v_max) * (1 - x) * y +
           this->dx(lvl, u_max, v_min) * x * (1 - y) +
           this->dx(lvl, u_min, v_min) * (1 - x) * (1 - y);
}

inline float ImagePyramid::mag2(const int lvl, const int u, const int v) const {
    return this->mag2_pyramid[lvl][idx(cam_data->width[lvl], u, v)];
}
inline int ImagePyramid::lvls() const {
    return image_pyramid.size();
}

template <typename T>
inline T halfSampling(const int child_u, const int child_v,
                      const int parent_step, const T* parent_image) {
    float parent_x = (child_u + 0.5f) * 2;  // x,y is in continue space
    float parent_y = (child_v + 0.5f) * 2;

    int parent_u_lo = parent_x - 0.5f;
    int parent_u_hi = parent_x + 0.5f;
    int parent_v_lo = parent_y - 0.5f;
    int parent_v_hi = parent_y + 0.5f;
    // todo getIdx(u,v,step)
    return 0.25 * (parent_image[idx(parent_step, parent_u_lo, parent_v_lo)] +
                   parent_image[idx(parent_step, parent_u_lo, parent_v_hi)] +
                   parent_image[idx(parent_step, parent_u_hi, parent_v_lo)] +
                   parent_image[idx(parent_step, parent_u_hi, parent_v_hi)]);
}

inline int idx(const int step, const int u, const int v) {
    return u + v * step;
}

}  // namespace mpl
