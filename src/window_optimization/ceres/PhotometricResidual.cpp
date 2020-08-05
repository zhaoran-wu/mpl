#include "PhotometricResidual.h"
#include "FrameParameterBlock.h"
#include "PhotometricBA.h"
#include "PointParameterBlock.h"
#include "candidate_manager.h"
#include "config.h"
#include "frame.h"
#include "pattern.h"

namespace mpl {
// help function

template <typename T>
bool project_to_target_frame(T uj, T vj, T iDepth, const Eigen::Matrix<T, 3, 3>& K, int width, int height,
                             const Eigen::Matrix<T, 3, 3>& R, const Eigen::Matrix<T, 3, 1>& t,
                             Eigen::Matrix<T, 2, 1>& Xpn, Eigen::Matrix<T, 2, 1>& hit_pixel, T& newIDepth,
                             T& ratio_inv_d);

template <typename T>
bool project_to_target_frame(T uj, T vj, T iDepth, int width, int height, const Eigen::Matrix<T, 3, 3>& KRKinv,
                             const Eigen::Matrix<T, 3, 1>& Kt, Eigen::Matrix<T, 2, 1>& hit_pixel);

// Cost Function

PhotometricCostFunction::PhotometricCostFunction(PhotometricResidual* const residual) : residual_(residual) {
    // number of residual
    this->set_num_residuals(PATTERN_SIZE);

    // parameter block sizes
    this->mutable_parameter_block_sizes()->push_back(9);  // owner frame: pose(7) + light(2)
    this->mutable_parameter_block_sizes()->push_back(9);  // target frame: pose(7) + light(2)
    this->mutable_parameter_block_sizes()->push_back(1);  // point inverse depth(1)
}

PhotometricCostFunction::~PhotometricCostFunction() {
}

bool PhotometricCostFunction::Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
    Candidate* point = this->residual_->point();
    Frame::ptr host_frame = this->residual_->host_frame();
    Frame::ptr obs_frame = this->residual_->obs_frame();

    const auto& config = Config::getInstance();
    const auto& cam = CamData::getInstance();

    const Eigen::Matrix3d& K = cam.K[0].cast<double>();
    const Eigen::Matrix3d& Kinv = cam.K_inv[0].cast<double>();
    const int32_t width = cam.width[0];
    const int32_t height = cam.height[0];

    const float energyThreshold = 1e10;  // TODO

    const auto& map_color = point->synetic_color;

    // residual computation
    // map pointer for efficiency. this avoids copying data
    Eigen::Map<Sophus::SE3d const> const T_w_h(parameters[0]);
    Eigen::Map<Sophus::SE3d const> const T_w_o(parameters[1]);

    const Sophus::SE3d T_o_w = T_w_o.inverse();
    const Sophus::SE3d T_o_h = T_o_w * T_w_h;
    const Eigen::Matrix3d R = T_o_h.rotationMatrix();
    const Eigen::Vector3d& t = T_o_h.translation();

    // affine light
    const AffineLight A_w_h((float)parameters[0][7], (float)parameters[0][8]);
    const AffineLight A_w_o((float)parameters[1][7], (float)parameters[1][8]);

    // inverse depth
    const double iDepth = parameters[2][0];

    // obtain geometric jacobians(T,d) for central pixel
    Eigen::Matrix<double, 2, 6> J_p_T_o_h;
    Eigen::Matrix<double, 2, 1> J_p_d;
    if (jacobians != NULL) {
        Eigen::Vector2d Xpn, hit_pixel;
        double newIDepth, ratio_inv_d;

        // project using central pixel
        if (!project_to_target_frame((double)point->u, (double)point->v, iDepth, K, width, height, R, t, Xpn, hit_pixel,
                                     newIDepth, ratio_inv_d)) {
            // discard this residual
            this->residual_->state_ = Visibility::OOB;
            this->discardOOB(residuals, jacobians);

            return true;
        }

        // inverse depth jacobian
        if (jacobians[2] != NULL) {
            J_p_d(0, 0) = K(0, 0) * ratio_inv_d * (t[0] - t[2] * Xpn[0]);
            J_p_d(1, 0) = K(1, 1) * ratio_inv_d * (t[1] - t[2] * Xpn[1]);
        }

        // pose jacobian
        if (jacobians[0] != NULL || jacobians[1] != NULL) {
            // relative pose jacobian
            J_p_T_o_h(0, 0) = K(0, 0) * newIDepth;
            J_p_T_o_h(0, 1) = 0.0;
            J_p_T_o_h(0, 2) = -K(0, 0) * newIDepth * Xpn[0];
            J_p_T_o_h(0, 3) = -K(0, 0) * Xpn[0] * Xpn[1];
            J_p_T_o_h(0, 4) = K(0, 0) * (1.0 + Xpn[0] * Xpn[0]);
            J_p_T_o_h(0, 5) = -K(0, 0) * Xpn[1];

            J_p_T_o_h(1, 0) = 0.0;
            J_p_T_o_h(1, 1) = K(1, 1) * newIDepth;
            J_p_T_o_h(1, 2) = -K(1, 1) * newIDepth * Xpn[1];
            J_p_T_o_h(1, 3) = -K(1, 1) * (1.0 + Xpn[1] * Xpn[1]);
            J_p_T_o_h(1, 4) = K(1, 1) * Xpn[0] * Xpn[1];
            J_p_T_o_h(1, 5) = K(1, 1) * Xpn[0];

            // adjoint
            J_p_T_o_h *= T_o_w.Adj();
        }
    }

    // compute residual for all pattern points
    /*     double energy = 0.0;
        double hessian = 0.0;
        double gradient = 0.0;
        double avgWeight = 0.0;
        double numBad = 0.0; */

    const Eigen::Matrix3d KRKinv = K * R * Kinv;
    const Eigen::Vector3d Kt = K * t;

    for (int32_t idx = 0; idx < PATTERN_SIZE; ++idx) {
        double uj = (double)point->u + (double)pattern[idx][0];
        double vj = (double)point->v + (double)pattern[idx][1];

        Eigen::Vector2d hit_pixel;

        if (!project_to_target_frame(uj, vj, iDepth, width, height, KRKinv, Kt, hit_pixel)) {
            // discard this residual
            this->residual_->state_ = Visibility::OOB;
            this->discardOOB(residuals, jacobians);

            return true;
        }

        // obtain bilinear interpolated intensity values
        double hit_color = obs_frame->at((float)hit_pixel(0), (float)hit_pixel(1));

        // residual with light compensation and weight
        double res = calc_light_diff((double)map_color[idx], hit_color, A_w_o.inverse());

        // image jacobian: dIj/du'
        Eigen::Matrix<double, 1, 2> J_r_p;
        J_r_p(0, 0) = obs_frame->dx((float)hit_pixel(0), (float)hit_pixel(1));
        J_r_p(0, 1) = obs_frame->dy((float)hit_pixel(0), (float)hit_pixel(1));

        double squaredGrad = J_r_p.squaredNorm();

        // observation weight based on gradient magnitude
        // set as the mean value between owner and target frames
        double gradWeight = std::sqrt(2500.f / (2500.f + squaredGrad));

        // apply gradient weight first
        const double resgw = res * gradWeight;

        // weight function
        const double totalWeight = gradWeight * point->weight[idx];

        // energy
        residuals[idx] = totalWeight * res;
        // energy += std::pow(totalWeight, 2) * res * res;

        // sum of gradients to avoid residuals in white walls
        // gradient += gradWeight * gradWeight * squaredGrad;

        // discard bad pixels
        if (std::abs(resgw) > energyThreshold) {
            std::cout << "resgw : " << resgw << '\n';
            // numBad++;
        }

        // compute jacobian
        if (jacobians != NULL) {
            // E = (1+alpha_i)*I_i(u) + beta_i - (1+alpha_j)*I(u') - beta_j
            // u' = project(X') = project(Tji * X) = project(Tji * unproject(u,
            // iDepth) ) Tji = Tj^-1 * Ti

            // jacobians[i][r * parameter_block_sizes_[i] + c]
            // i: parameter block
            // r: residual
            // c: parameter from block

            const Eigen::Matrix<double, 1, 6> J_r_T_o_h = totalWeight * J_r_p * J_p_T_o_h;
            const double J_r_A_o_h = -totalWeight * exp(A_w_o.inverse().alpha()) * map_color[idx];

            // host frame: J1x9
            if (jacobians[0] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 1, 9>> J_host(jacobians[0] + (idx * 9));
                J_host.segment<6>(0) = J_r_T_o_h;  // host pose
                J_host[6] = 0.0f;                  // owner alpha
                J_host[7] = 0.0f;                  // owner beta
                J_host[8] = 0.0;

                // variable scaling
                // "Numerical Optimization" Nocedal et al. 2006, page 95
                J_host.segment<3>(0) *= config.OPTIMIZATION_TRANS_SCALE;
                J_host.segment<3>(3) *= config.OPTIMIZATION_ROTATION_SCALE;
                J_host[6] *= config.OPTIMIZATION_ALPHA_SCALE;
                J_host[7] *= config.OPTIMIZATION_BETA_SCALE;

                // LOG(INFO) << "Jowner" << '\n' << J_host << '\n';
            }

            // target frame: J1x9
            if (jacobians[1] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 1, 9>> J_obs(jacobians[1] + (idx * 9));
                J_obs.segment<6>(0) = -J_r_T_o_h;  // target pose
                J_obs[6] = J_r_A_o_h;              // target alpha
                J_obs[7] = -1;                     // target beta
                J_obs[8] = 0.0;

                // variable scaling
                // "Numerical Optimization" Nocedal et al. 2006, page 95
                J_obs.segment<3>(0) *= config.OPTIMIZATION_TRANS_SCALE;
                J_obs.segment<3>(3) *= config.OPTIMIZATION_ROTATION_SCALE;
                J_obs[6] *= config.OPTIMIZATION_ALPHA_SCALE;
                J_obs[7] *= config.OPTIMIZATION_BETA_SCALE;
                // LOG(INFO) << "Jtarget" << '\n' << J_obs << '\n' << '\n';
            }

            // point inverse depth: J1x1
            if (jacobians[2] != NULL) {
                double iDepthJacob = totalWeight * J_r_p * J_p_d;

                jacobians[2][idx] = iDepthJacob * config.OPTIMIZATION_IDEPTH_SCALE;

                // hessian += iDepthJacob * iDepthJacob;
            }
        }
    }

    // check outlier
    //! if ((float)numBad / PATTERN_SIZE > 0.5 || gradient < 10.0) {
    //!     this->residual_->state_ = Visibility::OUTLIER;
    //!     // std::cout << "bad ratio:" << (float)numBad / PATTERN_SIZE
    //!     //          << "   gradient: " << gradient << '\n';
    //!     this->discardOutlier(jacobians);
    //! } else {
    //!     this->residual_->state_ = Visibility::VISIBLE;
    //! }

    // store useful values
    // this->residual_->energy_ = energy;
    // this->residual_->lossWeight_ = avgWeight / PATTERN_SIZE;

    //!    if (jacobians != NULL && jacobians[2] != NULL) {
    //!        this->residual_->iDepthHessian_ = hessian;
    //!    }

    return true;
}

void PhotometricCostFunction::discardOOB(double* residuals, double** jacobians) const {
    const int numPoints = PATTERN_SIZE;

    std::fill(residuals, residuals + numPoints, 0.0);

    if (jacobians != NULL) {
        // all jacobians to zero
        if (jacobians[0] != NULL) {
            std::fill(jacobians[0], jacobians[0] + (numPoints * 9), 0.0);
        }
        if (jacobians[1] != NULL) {
            std::fill(jacobians[1], jacobians[1] + (numPoints * 9), 0.0);
        }
        if (jacobians[2] != NULL) {
            std::fill(jacobians[2], jacobians[2] + numPoints, 0.0);
        }
    }
}

void PhotometricCostFunction::discardOutlier(double** jacobians) const {
    if (jacobians != NULL) {
        const int numPoints = PATTERN_SIZE;

        // all jacobians to zero
        if (jacobians[0] != NULL) {
            std::fill(jacobians[0], jacobians[0] + (numPoints * 9), 0.0);
        }
        if (jacobians[1] != NULL) {
            std::fill(jacobians[1], jacobians[1] + (numPoints * 9), 0.0);
        }
        if (jacobians[2] != NULL) {
            std::fill(jacobians[2], jacobians[2] + numPoints, 0.0);
        }
    }
}

void PhotometricCostFunction::discardOutlier(double** jacobians, int idx) const {
    if (jacobians != NULL) {
        if (jacobians[0] != NULL) {
            Eigen::Map<Eigen::Matrix<double, 1, 9>> Jowner(jacobians[0] + (idx * 9));
            Jowner.setZero();
        }
        if (jacobians[1] != NULL) {
            Eigen::Map<Eigen::Matrix<double, 1, 9>> Jtarget(jacobians[1] + (idx * 9));
            Jtarget.setZero();
        }
        if (jacobians[2] != NULL) {
            jacobians[2][idx] = 0.0;
        }
    }
}

// Residual

PhotometricResidual::PhotometricResidual(Candidate* point, const std::shared_ptr<Frame>& targetFrame)
    : point_(point), host_frame_(point->host_frame), target_frame_(targetFrame) {
    this->state_ = Visibility::VISIBLE;

    this->iDepthHessian_ = 0.0;
    this->energy_ = -1.0;
    this->lossWeight_ = 1.0;

    this->pixelEnergy_.resize(PATTERN_SIZE);
    this->pixelEnergy_.fill(-1);

    this->costFunction_ = std::make_unique<PhotometricCostFunction>(this);
}

PhotometricResidual::~PhotometricResidual() {
}

int PhotometricResidual::dimension() const {
    return this->costFunction_->num_residuals();
}

int PhotometricResidual::numParameterBlocks() const {
    return (int)this->costFunction_->parameter_block_sizes().size();
}

ceres::CostFunction* PhotometricResidual::getCostFunction() const {
    return this->costFunction_.get();
}

Candidate* PhotometricResidual::point() const {
    return this->point_;
}

Frame::ptr PhotometricResidual::host_frame() const {
    return this->host_frame_;
}

Frame::ptr PhotometricResidual::obs_frame() const {
    return this->target_frame_;
}

Visibility PhotometricResidual::state() const {
    return this->state_;
}

double PhotometricResidual::iDepthHessian() const {
    return this->iDepthHessian_;
}

double PhotometricResidual::energy() const {
    return this->energy_;
}

const Eigen::VectorXd& PhotometricResidual::pixelEnergy() const {
    return this->pixelEnergy_;
}

double PhotometricResidual::lossWeight() const {
    return this->lossWeight_;
}

/* bool PhotometricResidual::evaluate(int lvl, Eigen::VecXf& residuals) const {
    const auto& settings = Settings::getInstance();

    if (!this->point_->valid(lvl)) {
        return false;
    }

    // output residuals
    residuals.resize(PATTERN_SIZE);

    // calibration
    const auto& calib = GlobalCalibration::getInstance();
    const Eigen::Matrix3f& K = calib.matrix3f(lvl);
    const Eigen::Matrix3f& Kinv = calib.invMatrix3f(lvl);
    const int32_t width = calib.width(lvl);
    const int32_t height = calib.height(lvl);

    // parameters
    const float iDepth = (float)this->point_->pointBlock()->getIDepth();
    const Sophus::SE3f ownerToWorld =
        this->ownerFrame_->get_frame_block()->getPose().cast<float>();
    const Sophus::SE3f targetToWorld =
        this->targetFrame_->get_frame_block()->getPose().cast<float>();
    const AffineLight ownerLight =
        this->ownerFrame_->get_frame_block()->getAffineLight();
    const AffineLight targetLight =
        this->targetFrame_->get_frame_block()->getAffineLight();

    const Sophus::SE3f worldToTarget = targetToWorld.inverse();
    const Sophus::SE3f ownerToTarget = worldToTarget * ownerToWorld;
    const Eigen::Matrix3f R = ownerToTarget.rotationMatrix();
    const Eigen::Vector3f& t = ownerToTarget.translation();

    const Eigen::Matrix3f KRKinv = K * R * Kinv;
    const Eigen::Vector3f Kt = K * t;

    const AffineLight relLight =
        AffineLight::calcRelative(ownerLight, targetLight);
    const float light_a = relLight.a();
    const float light_b = relLight.b();

    // reference
    const Eigen::VecXf& color = this->point_->colors(lvl);
    const Eigen::VecXf& weight = this->point_->weights(lvl);

    const float* image = this->targetFrame_->image(lvl);
    const float* gx = this->targetFrame_->gx(lvl);
    const float* gy = this->targetFrame_->gy(lvl);

    for (int32_t idx = 0; idx < PATTERN_SIZE; ++idx) {
        const float uj = this->point_->u(lvl) + (float)Pattern::at(idx, 0);
        const float vj = this->point_->v(lvl) + (float)Pattern::at(idx, 1);

        Eigen::Vector2f hit_pixel;
        if (!project_to_target_frame(uj, vj, iDepth, width, height, KRKinv, Kt,
                                     hit_pixel)) {
            return false;
        }

        // obtain bilinear interpolated intensity values
        const float hit_color =
            bilinearInterpolation(image, hit_pixel[0], hit_pixel[1], width);

        // residual with light compensation
        const float res = color[idx] - light_a * hit_color - light_b;

        // image gradient: dIj/du'
        Eigen::Matrix<float, 1, 2> J_r_p;
        J_r_p(0, 0) = light_a * bilinearInterpolation(gx, hit_pixel[0],
                                                      hit_pixel[1], width);
        J_r_p(0, 1) = light_a * bilinearInterpolation(gy, hit_pixel[0],
                                                      hit_pixel[1], width);

        // observation weight based on gradient magnitude
        // set as the mean value between owner and target frames
        float gradWeight =
            sqrt(settings.weightConstant /
                 (settings.weightConstant + J_r_p.squaredNorm()));
        gradWeight = 0.5f * (weight[idx] + gradWeight);

        // apply gradient weight first
        residuals[idx] = res * gradWeight;
    }

    return true;
} */

// Projection functions
// We will use inverse depth parameterization to achieve a continuous
// parameterization. In addition, we scale the transformation function to get a
// continuous function:
//
//		hit_pixel = K*R*inv(K)*u + K*t*iDepth
//
// This scaled function is continuous and avoids computing 3D points, which will
// lead to a discontinuity in iDepth = 0. We will also use this function during
// jacobians estimation to make sure they are continuous in all the values of
// the inverse depth.

// This function transforms a pixel from the reference image to a new one.
// It also checks image boundaries and inverse depth consistency even if
// its values is < 0. Its output contains useful values for jacobians
// computation.
//
// in - uj: pixel x coordinate
// in - vj: pixel y coordinate
// in - iDepth: pixel inverse depth
// in - normal: point normal, or camera ray
// in - K: camera intrinsic matrix
// in - width, height: image dimensions
// in - R: rotation from reference to new image
// in - t: translation from reference to new image
// out - Xpn: Xp normalized at z=1
// out - hit_pixel: projected point in new image
// out - newIDepth: inverse depth in new image
// out - ratio_inv_d: scale between both inverse depths (newIDepth / iDepth)
// return: if successfully projected or not due to OOB
template <typename T>
inline bool project_to_target_frame(T uj, T vj, T iDepth, const Eigen::Matrix<T, 3, 3>& K, int width, int height,
                                    const Eigen::Matrix<T, 3, 3>& R, const Eigen::Matrix<T, 3, 1>& t,
                                    Eigen::Matrix<T, 2, 1>& Xpn, Eigen::Matrix<T, 2, 1>& hit_pixel, T& newIDepth,
                                    T& ratio_inv_d) {
    auto& cam = CamData::getInstance();

    // unproject
    const Eigen::Matrix<T, 3, 1> Xinv((uj - K(0, 2)) / K(0, 0), (vj - K(1, 2)) / K(1, 1), 1);

    // Xp: R*Xinv + t*iDepth
    const Eigen::Matrix<T, 3, 1> Xp = R * Xinv + t * iDepth;

    // new iDepth and ratio_inv_d factor
    ratio_inv_d = 1 / Xp[2];

    // if the point was in the range [0, Inf] in camera1
    // it has to be also in the same range in camera2
    // This allows using negative inverse depth values
    // i.e. same iDepth sign in both cameras
    if (!(ratio_inv_d > 0)) return false;

    // inverse depth in new image
    newIDepth = iDepth * ratio_inv_d;

    // normalize
    Xpn[0] = Xp[0] * ratio_inv_d;
    Xpn[1] = Xp[1] * ratio_inv_d;

    // project: K * Xp
    hit_pixel[0] = Xpn[0] * K(0, 0) + K(0, 2);
    hit_pixel[1] = Xpn[1] * K(1, 1) + K(1, 2);

    // check image boundaries
    return is_in_img(cam, hit_pixel);
}

// This function transforms a pixel from the reference image to a new one.
// It also checks image boundaries and inverse depth consistency even if
// its values is < 0.
//
// in - uj: pixel x coordinate
// in - vj: pixel y coordinate
// in - iDepth: pixel inverse depth
// in - width, height: image dimensions
// in - KRKinv: K*rotation*inv(K) from reference to new image
// in - Kt: K*translation from reference to new image
// out - hit_pixel: projected point in new image
// return: if successfully projected or not due to OOB
template <typename T>
inline bool project_to_target_frame(T uj, T vj, T iDepth, int width, int height, const Eigen::Matrix<T, 3, 3>& KRKinv,
                                    const Eigen::Matrix<T, 3, 1>& Kt, Eigen::Matrix<T, 2, 1>& hit_pixel) {
    auto& cam = CamData::getInstance();
    // transform and project
    const Eigen::Matrix<T, 3, 1> pt = KRKinv * Eigen::Matrix<T, 3, 1>(uj, vj, 1) + Kt * iDepth;

    // ratio_d_inv factor
    const T ratio_d_inv = 1 / pt[2];

    // if the point was in the range [0, Inf] in camera1
    // it has to be also in the same range in camera2
    // This allows using negative inverse depth values
    // i.e. same iDepth sign in both cameras
    if (!(ratio_d_inv > 0)) return false;

    // normalize
    hit_pixel[0] = pt[0] * ratio_d_inv;
    hit_pixel[1] = pt[1] * ratio_d_inv;

    // check image boundaries
    return is_in_img(cam, hit_pixel);
}

}  // namespace mpl