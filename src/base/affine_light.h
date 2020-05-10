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

    // irrandeance = exp(-alpha) * (I_cam - beta)
    float alpha() const;
    float beta() const;

    Eigen::Vector2f getEigenVec() const;

    // I = a*irrandeance +b
    float a() const;
    float b() const;

    /**
     * @brief   calculate two relative affinelight between two frames
    * intput 4 global param: (alpha_src,beta_src,alpha_dst,beta_dst)
    * !exp(-alpha_dst)*(I_dst-beta_dst) = exp(-alpha_src)*(I_src - beta_src)

    * output 2 relative param (a,b): I_dst = exp(alpha_dst - alpha_src)* I_src
    * + beta_dst - beta_src*exp(alpha_dst -alpha_src)
    * I_dst = exp(alpha)*I_src + beta
    * so we will have the relation ship beteween relative and global affine
    * param
    //* alpha = alpha_dst - alpha_src
    //* beta = beta_dst - exp(alpha)*beta_src
    //* the output affine param map src I to dst I
     *
     * @param src_global src affine light parameter,map I_src to Irran_src
     * @param dst_global dst affine light parameter,map I_dst to Irran_dst
     * @return relative AffineLight map intensity of src to intensity of dst
     *  I_dst = exp(result.alpha)* I_src + result.beta
     */
    static AffineLight calc_aff_map_src_to_dst(const AffineLight& src,
                                               const AffineLight& dst);

    /**
     * @brief we want to calc target global affind param, with already known
     * relative to src
     *
     * @param src src affine light parameter,map I_src to Irran_src
     * @param dst_src relative affine light param, map I_src to I_dst
     * @return dst affine light param ,map I_dst to Irran_dst
     */
    static AffineLight calc_dst_global_aff(const AffineLight& src,
                                           const AffineLight& dst_src);

   private:
    float alpha_;
    float beta_;
};

}  // namespace mpl