#include "../util/utility.h"
#include "cam_data.h"
#include "config.h"
#include "image_pyramid.h"
#include "pixel_selector.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sophus/se3.hpp>
#include <vector>

const float cx = 325.5;
const float cy = 253.5;
const float fx = 518.0;
const float fy = 519.0;
template <typename T>
T getPixelValue(double x, double y, const cv::Mat* im) {
    T* data = &im->data[int(y) * im->step + int(x)];
    double xx = x - floor(x);
    double yy = y - floor(y);
    return T((1 - xx) * (1 - yy) * data[0] + xx * (1 - yy) * data[1] +
             (1 - xx) * yy * data[im->step] + xx * yy * data[im->step + 1]);
}
inline Eigen::Vector3d project2Dto3D(int x, int y, ushort d, float fx, float fy,
                                     float cx, float cy, float scale) {
    float zz = float(d) / scale;
    float xx = zz * (x + 0.5f - cx) / fx;
    float yy = zz * (y + 0.5f - cy) / fy;
    return Eigen::Vector3d(xx, yy, zz);
}

inline cv::Point2d project3Dto2D(float x, float y, float z, float fx, float fy,
                                 float cx, float cy) {
    double u = fx * x / z + cx;
    double v = fy * y / z + cy;
    return cv::Point2d(u, v);
}

class PhotometicCostFunction : public ceres::SizedCostFunction<1, 9> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    PhotometicCostFunction(Eigen::Vector3d point_, double intensity1_,
                           double fx, double fy, double cx, double cy,
                           cv::Mat im2_, cv::Mat grad2x_, cv::Mat grad2y_)
        : point(point_),
          intensity1(intensity1_),
          fx(fx),
          fy(fy),
          cx(cx),
          cy(cy),
          im2(im2_),
          grad2_x(grad2x_),
          grad2_y(grad2y_) {
    }

    virtual ~PhotometicCostFunction() = default;

    // params :
    //      [q,t,a,b]
    virtual bool Evaluate(double const* const* parameters, double* residuals,
                          double** jacobians) const override;

   private:
    const Eigen::Vector3d point;  // 3d point in the im1 ccs
    const double intensity1;      // intensity in im1
    const double fx, fy, cx, cy;
    const cv::Mat im2;
    const cv::Mat grad2_x, grad2_y;  // gradient image in x and y direction
};
// implementation
bool PhotometicCostFunction::Evaluate(double const* const* parameters,
                                      double* residuals,
                                      double** jacobians) const {
    //* #######---compute photometic residual----######
    Eigen::Map<const Sophus::SE3d> const T(parameters[0]);
    Eigen::Vector3d p_3d = T * point;  // T21
    cv::Point2d p_2d = project3Dto2D(p_3d(0), p_3d(1), p_3d(2), fx, fy, cx, cy);

    if (p_2d.x < 0 || p_2d.x > im2.cols || p_2d.y < 0 || p_2d.y > im2.rows) {
        // residuals[0] = 0;
        if (jacobians != NULL && jacobians[0] != NULL) {
            std::fill(&jacobians[0][0], &jacobians[0][9], 0);  //! over last
        }
        residuals[0] = 0;
        return true;
    }
    residuals[0] = -exp(parameters[0][7]) * intensity1 - parameters[0][8] +
                   getPixelValue<uchar>(p_2d.x, p_2d.y, &im2);
    //* #######---set Jacobdians J = [JT,Jab], with dimension 1*(6+2)----######
    // JT = dI/dT = dI/dp * dp / dT
    Eigen::Matrix<double, 1, 2> dIdp;
    dIdp(0, 0) = grad2_x.at<float>(p_2d);
    dIdp(0, 1) = grad2_y.at<float>(p_2d);
    // dp/dT 2*6
    Eigen::Matrix<double, 2, 6> dpdT;
    double x = p_3d(0);
    double y = p_3d(1);
    double z = p_3d(2);
    dpdT(0, 0) = fx / z;
    dpdT(0, 1) = 0;
    dpdT(0, 2) = -fx * x / z * z;
    dpdT(0, 3) = -fx * x * y / z * z;
    dpdT(0, 4) = fx * (1 + x * x / z * z);
    dpdT(0, 5) = -fx * y / z;

    dpdT(1, 0) = 0;
    dpdT(1, 1) = fy / z;
    dpdT(1, 2) = -fy * y / z * z;
    dpdT(1, 3) = -fy * (1 + y * y / z * z);
    dpdT(1, 4) = fy * x * y / z * z;
    dpdT(1, 5) = fy * x / z;

    if (jacobians != NULL && jacobians[0] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 1, 6>> JT(jacobians[0]);
        JT = dIdp * dpdT;
        JT.segment<3>(0) *= 0.5f;
        jacobians[0][6] = -10 * intensity1 * exp(parameters[0][7]);
        jacobians[0][7] = -1000 * 1.0;
        jacobians[0][8] = 0;
    }

    return true;
}

class Parameterization : public ceres::LocalParameterization {
   public:
    Parameterization() = default;
    virtual ~Parameterization() = default;

    virtual bool Plus(const double* x, const double* delta,
                      double* x_plus_delta) const override;

    virtual bool ComputeJacobian(const double* x,
                                 double* jacobian) const override {
        Eigen::Map<Eigen::Matrix<double, 9, 8, Eigen::RowMajor>> J(jacobian);
        J.setZero();
        J.block<8, 8>(0, 0).setIdentity();
        return true;
    }

    virtual int GlobalSize() const override {
        return 9;
    }

    virtual int LocalSize() const override {
        return 8;
    }
};
/**
 *   implementation
 *
 */
bool Parameterization::Plus(const double* x, const double* delta,
                            double* x_plus_delta) const {
    const Eigen::Map<const Sophus::SE3d> T(x);
    const Eigen::Map<const Sophus::Vector6d> delta_se3(delta);
    Eigen::Map<Sophus::SE3d> T_new(x_plus_delta);

    //! we drive the jacobian with the sequence of (pho,phi);
    //! so the delta_se3 will be (pro,phi)

    //* #######---update T ----######
    Sophus::Vector6d pose_delta;
    pose_delta.segment<3>(0) += 0.5 * delta_se3.segment<3>(0);
    pose_delta.segment<3>(3) += delta_se3.segment<3>(3);

    //! the first 3 is translational in sohpus (pho phi), determined by jacobian
    T_new = Sophus::SE3d::exp(pose_delta) * T;

    x_plus_delta[7] = x[7] + delta[6] * 10;
    x_plus_delta[8] = x[8] + delta[7] * 1000;

    return true;
}

Eigen::Vector3d projectTo3d(const cv::Point pixel, const Eigen::Matrix3d K,
                            const cv::Mat depth) {
    Eigen::Vector3d homo_pixel(pixel.x, pixel.y, 1);
    /*   float d = depth.at<uchar>(pixel) * 40 /
                255.0f;  //100; //! 40 is z_far //255 -> 256 faster */
    float d = depth.at<ushort>(pixel) * 0.001;
    // LOG(INFO) << " depth " << d << " m";
    return d * K.inverse() * homo_pixel;
}
using namespace mpl;
int main() {
    Config& config = Config::getInstance();
    std::string config_path = "/home/zhaoran/thesis_ws/mpl/project/config.yaml";
    config.readYamlFile(config_path);

    std::string calib_path =
        "/home/zhaoran/thesis_ws/mpl/project/camera_tum.yaml";
    CamData& cam = CamData::getInstance();
    cam.readYamlFile(calib_path);
    const std::string im_path = "/home/zhaoran/thesis_ws/mpl/test_data/";
    //* all image
    cv::Mat im1 = cv::imread(im_path + "rgb/1.png");
    cv::Mat im1_clone = im1.clone();
    cv::cvtColor(im1, im1, CV_BGR2GRAY);

    cv::Mat im2 = cv::imread(im_path + "rgb/2.png");
    cv::Mat im2_clone = im2.clone();
    cv::cvtColor(im2, im2, CV_BGR2GRAY);

    cv::Mat depth1 = cv::imread(im_path + "depth/1.png");

    std::shared_ptr<ImagePyramid> pyramid(new ImagePyramid(im1.data));
    //* compute gradient image
    cv::Mat grad2x(im2.size(), CV_32F, pyramid->dx(0).get());
    cv::Mat grad2y(im2.size(), CV_32F, pyramid->dy(0).get());

    std::vector<cv::Point> candidates;

    PixelSelector selector;
    std::vector<Eigen::Vector3i> candidates_eigen;
    selector.select(pyramid, candidates_eigen);

    for (const auto& p : candidates_eigen) {
        candidates.push_back(cv::Point(p(0), p(1)));
    }

    //* build optimization probleml
    ceres::Problem problem;
    double* param = new double[9];
    Eigen::Map<Sophus::SE3d> T(param);
    T = Sophus::SE3d(Eigen::Quaterniond::Identity(),
                     Eigen::Vector3d(-0.00, -0.00, 0));  //! T21
    param[7] = 0;
    param[8] = 0;

    ceres::LocalParameterization* parameterization = new Parameterization();
    ceres::LossFunction* loss = NULL;  // new ceres::HuberLoss(30.0);
    // solve to get the pose
    for (const auto& pixel : candidates) {
        if (depth1.at<ushort>(pixel) == 0) continue;

        Eigen::Vector3d p_3d1 = project2Dto3D(
            pixel.x, pixel.y, depth1.at<ushort>(pixel), fx, fy, cx, cy, 1000);
        ceres::CostFunction* cost = new PhotometicCostFunction(
            p_3d1, im1.at<uchar>(pixel), fx, fy, cx, cy, im2, grad2x, grad2y);
        problem.AddResidualBlock(cost, loss, param);
        problem.SetParameterization(param, parameterization);
    }
    // before
    for (const auto& p : candidates) {
        if (depth1.at<ushort>(p) == 0) continue;
        cv::circle(im1_clone, p, 1, cv::Scalar(0, 255, 0), 1, CV_FILLED);
        Eigen::Vector3d p_3 =
            project2Dto3D(p.x, p.y, depth1.at<ushort>(p), fx, fy, cx, cy, 1000);
        Eigen::Vector3d p_x = p_3;
        cv::Point2d p_2 = project3Dto2D(p_x(0), p_x(1), p_x(2), fx, fy, cx, cy);
        cv::Point p_draw(p_2.x, p_2.y);
        cv::circle(im2_clone, p_draw, 1, cv::Scalar(0, 255, 0), 1, CV_FILLED);
    }
    cv::imshow("im2_clone ", im2_clone);
    cv::imshow("im1_clone ", im1_clone);
    cv::waitKey(0);

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = 1;
    options.update_state_every_iteration = true;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.max_num_iterations = 100;
    ceres::Solve(options, &problem, &summary);
    // options.linear_solver_type = ceres::DENSE_SCHUR;

    std::cout << summary.FullReport() << "\n";
    //* after
    for (size_t i = 0; i < 9; ++i) {
        LOG(ERROR) << "param" << i << " : " << param[i];
    }
    Eigen::Map<Sophus::SE3d> T_x(param);
    for (const auto& p : candidates) {
        if (depth1.at<ushort>(p) == 0) continue;
        cv::circle(im1_clone, p, 1, cv::Scalar(0, 255, 0), 1, CV_FILLED);
        Eigen::Vector3d p_3 =
            project2Dto3D(p.x, p.y, depth1.at<ushort>(p), fx, fy, cx, cy, 1000);
        Eigen::Vector3d p_x = T_x * p_3;
        cv::Point2d p_2 = project3Dto2D(p_x(0), p_x(1), p_x(2), fx, fy, cx, cy);
        cv::Point p_draw(p_2.x, p_2.y);
        cv::circle(im2_clone, p_draw, 1, cv::Scalar(0, 0, 255), 1, CV_FILLED);
    }
    cv::imshow("im2_clone ", im2_clone);
    cv::imshow("im1_clone ", im1_clone);
    cv::waitKey(0);

    return 0;
}