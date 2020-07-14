#pragma once
#include <Eigen/Core>
namespace mpl {
/**
 * @brief class to parameterize affine light
 *      I = irrandeance*t_exposure
 *
 *      model with affine model
 *      I = exp(alpha)*irrandeance + beta
 *      ! affineLight(alpha, beta) map irrandeance to intensity
 *      (irradeance = exp(-alpha)*(I - beta) = a* I + b);
 *      a,b use to map Intensity to Irrandiance
 *
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

    void update(float delta_alpha, float delta_beta);

    // irrandeance = exp(-alpha) * (I_cam - beta)
    float alpha() const;
    float beta() const;

    Eigen::Vector2f getEigenVec() const;

    // I = a*irrandeance +b
    float a() const;
    float b() const;

    // calc relative affine param,which map I_src to I_dst
    static AffineLight calc_aff_map_src_to_dst(const AffineLight& src,
                                               const AffineLight& dst);

    // calc absolute affine param of dst
    // dst_src: Aff param map src to dst
    static AffineLight calc_dst_global_aff(const AffineLight& src,
                                           const AffineLight& dst_src);
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

}  // namespace mpl