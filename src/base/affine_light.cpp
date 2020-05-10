#include "affine_light.h"

namespace mpl {

AffineLight::AffineLight() : alpha_(0.f), beta_(0.f) {
}

AffineLight::AffineLight(float alpha, float beta) : alpha_(alpha), beta_(beta) {
}

AffineLight::AffineLight(const Eigen::Vector2f& affLight)
    : alpha_(affLight[0]), beta_(affLight[1]) {
}

AffineLight::AffineLight(const AffineLight& affLight)
    : alpha_(affLight.alpha_), beta_(affLight.beta_) {
}

float AffineLight::alpha() const {
    return this->alpha_;
}

float AffineLight::beta() const {
    return this->beta_;
}

Eigen::Vector2f AffineLight::getEigenVec() const {
    return Eigen::Vector2f(this->alpha_, this->beta_);
}

float AffineLight::a() const {
    return exp(-alpha_);
}

float AffineLight::b() const {
    return -exp(-alpha_) * this->beta_;
}

AffineLight& AffineLight::operator=(const AffineLight& affLight) {
    this->alpha_ = affLight.alpha_;
    this->beta_ = affLight.beta_;
    return *this;
}

AffineLight AffineLight::calc_aff_map_src_to_dst(const AffineLight& src,
                                                 const AffineLight& dst) {
    // dst_src : affine parame map I_src to I_dst
    float dst_src_alpha = dst.alpha() - src.alpha();
    float dst_src_beta = dst.beta() - src.beta() * exp(dst_src_alpha);

    return AffineLight(dst_src_alpha, dst_src_beta);
}

AffineLight AffineLight::calc_dst_global_aff(const AffineLight& src,
                                             const AffineLight& dst_src) {
    float dst_alpha = dst_src.alpha() + src.alpha();
    float dst_beta = dst_src.beta() + src.beta() * exp(dst_src.alpha());

    return AffineLight(dst_alpha, dst_beta);
}

}  // namespace mpl