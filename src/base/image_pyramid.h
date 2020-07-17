#pragma once
#include "cam_data.h"
#include "config.h"
#include <iostream>
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
    float_ptr mag_squared(const int lvl);
    // get pixel value at lvl
    uchar at(const int u, const int v, const int lvl = 0) const;
    float dy(const int u, const int v, const int lvl = 0) const;
    float dx(const int u, const int v, const int lvl = 0) const;
    float mag_squared(const int u, const int v, const int lvl = 0) const;

    // sub-pixel overloading, u,v is in (continously) pixel coordiante system
    float at(const float u, const float v, const int lvl = 0) const;
    float dx(const float u, const float v, const int lvl = 0) const;
    float dy(const float u, const float v, const int lvl = 0) const;
    // todo overload for eigen vec
    bool is_in_image(const float u, const float v, const int lvl = 0) const;

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

inline ImagePyramid::float_ptr ImagePyramid::mag_squared(const int lvl) {
    return this->mag2_pyramid[lvl];
}

inline bool ImagePyramid::is_in_image(const float u, const float v, const int lvl) const {
    return (u > 2.f && u < cam_data->width[lvl] - 2.f && v > 2.f && v < cam_data->height[lvl] - 2.f);
}

inline uchar ImagePyramid::at(const int u, const int v, const int lvl) const {
    return this->image_pyramid[lvl][idx(cam_data->width[lvl], u, v)];
}

inline float ImagePyramid::dy(const int u, const int v, const int lvl) const {
    return this->dy_pyramid[lvl][idx(cam_data->width[lvl], u, v)];
}

inline float ImagePyramid::dx(const int u, const int v, const int lvl) const {
    return this->dx_pyramid[lvl][idx(cam_data->width[lvl], u, v)];
}

// u,v here is hit pixel position
inline float ImagePyramid::at(const float u, const float v, const int lvl) const {
    int u_max = std::ceil(u);
    int v_max = std::ceil(v);
    int u_min = u_max - 1;
    int v_min = v_max - 1;

    float x = u - u_min;
    float y = v - v_min;

    return this->at(u_max, v_max, lvl) * x * y + this->at(u_min, v_max, lvl) * (1 - x) * y +
           this->at(u_max, v_min, lvl) * x * (1 - y) + this->at(u_min, v_min, lvl) * (1 - x) * (1 - y);
}

inline float ImagePyramid::dy(const float u, const float v, const int lvl) const {
    int u_max = std::ceil(u);
    int v_max = std::ceil(v);
    int u_min = u_max - 1;
    int v_min = v_max - 1;

    float x = u - u_min;
    float y = v - v_min;

    float value = this->dy(u_max, v_max, lvl) * x * y + this->dy(u_min, v_max, lvl) * (1 - x) * y +
                  this->dy(u_max, v_min, lvl) * x * (1 - y) + this->dy(u_min, v_min, lvl) * (1 - x) * (1 - y);
    return value;
}
/**
 * @brief float version is for hit pixel, so the u,v are not in pixel coordinate
 *  but continous image coordinate
 *
 * @param lvl
 * @param u : continous image x coordinate
 * @param v  : continous image y coordinate
 * @return float
 */
inline float ImagePyramid::dx(const float u, const float v, const int lvl) const {
    int u_max = std::ceil(u);
    int v_max = std::ceil(v);
    int u_min = u_max - 1;
    int v_min = v_max - 1;

    float x = u - u_min;
    float y = v - v_min;

    float value = this->dx(u_max, v_max, lvl) * x * y + this->dx(u_min, v_max, lvl) * (1 - x) * y +
                  this->dx(u_max, v_min, lvl) * x * (1 - y) + this->dx(u_min, v_min, lvl) * (1 - x) * (1 - y);
    return value;
}

inline float ImagePyramid::mag_squared(const int u, const int v, const int lvl) const {
    return this->mag2_pyramid[lvl][idx(cam_data->width[lvl], u, v)];
}
inline int ImagePyramid::lvls() const {
    return image_pyramid.size();
}

template <typename T>
inline T halfSampling(const int child_u, const int child_v, const int parent_step, const T* parent_image) {
    int parent_u_lo = 2 * child_u;
    int parent_u_hi = parent_u_lo + 1;
    int parent_v_lo = 2 * child_v;
    int parent_v_hi = parent_v_lo + 1;
    return 0.25f * (parent_image[idx(parent_step, parent_u_lo, parent_v_lo)] +
                    parent_image[idx(parent_step, parent_u_lo, parent_v_hi)] +
                    parent_image[idx(parent_step, parent_u_hi, parent_v_lo)] +
                    parent_image[idx(parent_step, parent_u_hi, parent_v_hi)]);
}

inline int idx(const int step, const int u, const int v) {
    return u + v * step;
}

}  // namespace mpl
