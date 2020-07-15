#pragma once
#include <Eigen/Core>
namespace mpl {
/**
 * @brief class to parameterize affine light
 * absolute:
 *       I' = B*t_exposure, I' is corrected light, B is irrandeance
 *
 *      model with affine model,when t_exposure is unknown
 *      I' = exp(alpha)*B + beta
 *
 *      affineLight(alpha, beta) map irrandeance to corrected light
 *      (irradeance = exp(-alpha)*(I' - beta) = a* I' + b);
 *      a,b use to map I' ---> B
 *relative:
 *      I1' = exp(alpha)*I2' + beta
 *
 */
class AffineLight {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
   public:
    AffineLight();
    AffineLight(float alpha, float beta);
    AffineLight(const Eigen::Vector2f& affLight);
    AffineLight(const AffineLight& affLight);
    AffineLight& operator=(const AffineLight& affLight);

    // update with a update step in optimization
    void update(float delta_alpha, float delta_beta);

    // I_corrected = exp(alpha) * B + beta
    float alpha() const;
    float beta() const;

    // B = a*I_corrected +b
    float a() const;
    float b() const;

    // calc relative affine param,which map I_src to I_dst
    static AffineLight calc_aff_map_src_to_dst(const AffineLight& src, const AffineLight& dst);

    // calc absolute affine param of dst
    // dst_src: Aff param map src to dst
    static AffineLight calc_dst_global_aff(const AffineLight& src, const AffineLight& dst_src);

    static const int DoF = 2;
    static const int num_parameters = 2;

   private:
    float alpha_;
    float beta_;
};

inline void AffineLight::update(float delta_alpha, float delta_beta) {
    alpha_ += delta_alpha;
    beta_ += delta_beta;
}

template <typename T>
inline T map_src_to_dst(const AffineLight& dst_src, const T src_light) {
    return exp(dst_src.alpha()) * src_light + dst_src.beta();
}

template <typename T>
inline T calc_light_diff(const T src_light, const T dst_light, const AffineLight& dst_src) {
    return dst_light - map_src_to_dst(dst_src, src_light);
}

inline AffineLight::AffineLight() : alpha_(0.f), beta_(0.f) {
}

inline AffineLight::AffineLight(float alpha, float beta) : alpha_(alpha), beta_(beta) {
}

inline AffineLight::AffineLight(const Eigen::Vector2f& affLight) : alpha_(affLight[0]), beta_(affLight[1]) {
}

inline AffineLight::AffineLight(const AffineLight& affLight) : alpha_(affLight.alpha_), beta_(affLight.beta_) {
}

inline float AffineLight::alpha() const {
    return this->alpha_;
}

inline float AffineLight::beta() const {
    return this->beta_;
}

inline float AffineLight::a() const {
    return exp(-alpha_);
}

inline float AffineLight::b() const {
    return -exp(-alpha_) * this->beta_;
}

inline AffineLight& AffineLight::operator=(const AffineLight& affLight) {
    this->alpha_ = affLight.alpha_;
    this->beta_ = affLight.beta_;
    return *this;
}

inline AffineLight AffineLight::calc_aff_map_src_to_dst(const AffineLight& src, const AffineLight& dst) {
    float dst_src_alpha = dst.alpha() - src.alpha();
    float dst_src_beta = dst.beta() - src.beta() * exp(dst_src_alpha);

    return AffineLight(dst_src_alpha, dst_src_beta);
}

inline AffineLight AffineLight::calc_dst_global_aff(const AffineLight& src, const AffineLight& dst_src) {
    float dst_alpha = dst_src.alpha() + src.alpha();
    float dst_beta = dst_src.beta() + src.beta() * exp(dst_src.alpha());

    return AffineLight(dst_alpha, dst_beta);
}
}  // namespace mpl