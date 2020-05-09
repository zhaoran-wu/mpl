#pragma once
#include "cam_data.h"
#include <memory>
#include <vector>
namespace mpl {
typedef unsigned char uchar;

inline uchar halfSampling(const int child_u, const int child_v,
                          const int parent_step, const uchar* parent_image);

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

    ImagePyramid(const uchar* const row_data, const int lvls);

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

inline float ImagePyramid::mag2(const int lvl, const int u, const int v) const {
    return this->mag2_pyramid[lvl][idx(cam_data->width[lvl], u, v)];
}
inline int ImagePyramid::lvls() const {
    return image_pyramid.size();
}

inline uchar halfSampling(const int child_u, const int child_v,
                          const int parent_step, const uchar* parent_image) {
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
