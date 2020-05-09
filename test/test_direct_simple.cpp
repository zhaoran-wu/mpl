#include "../util/utility.h"
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
inline Eigen::Vector3d project2Dto3D(int x, int y, int d, float fx, float fy,
                                     float cx, float cy, float scale) {
    float zz = float(d) / scale;
    float xx = zz * (x + 0.5 - cx) / fx;
    float yy = zz * (y + 0.5 - cy) / fy;
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
    residuals[0] =
        intensity1 -
        exp(parameters[0][7]) * getPixelValue<uchar>(p_2d.x, p_2d.y, &im2) -
        parameters[0][8];
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
        JT = -dIdp * dpdT;
        jacobians[0][6] =
            -getPixelValue<uchar>(p_2d.x, p_2d.y, &im2) * exp(parameters[0][7]);
        jacobians[0][7] = -1;
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
    T_new = Sophus::SE3d::exp(delta_se3) * T;
    x_plus_delta[7] = x[7] + delta[6];
    x_plus_delta[8] = x[8] + delta[7];

    return true;
}

Eigen::Vector3d projectTo3d(const cv::Point pixel, const Eigen::Matrix3d K,
                            const cv::Mat depth) {
    Eigen::Vector3d homo_pixel(pixel.x, pixel.y, 1);
    /*   float d = depth.at<uchar>(pixel) * 40 /
                255.0f;  //*100; //! 40 is z_far //255 -> 256 faster */
    float d = depth.at<ushort>(pixel) * 0.001;
    // LOG(INFO) << " depth " << d << " m";
    return d * K.inverse() * homo_pixel;
}

int main(int argc, char** argv) {
    const std::string im_path = "/home/zhaoran/thesis_ws/mpl/test_data/";
    //* all image
    cv::Mat im1 = cv::imread(im_path + "tum.1.photo.png");
    cv::Mat im1_clone = im1.clone();
    cv::cvtColor(im1, im1, CV_BGR2GRAY);

    cv::Mat im2 = cv::imread(im_path + "tum.2.photo.png");
    cv::Mat im2_clone = im2.clone();
    cv::cvtColor(im2, im2, CV_BGR2GRAY);

    cv::Mat depth1 = cv::imread(im_path + "tum.1.depth.png");

    //* compute gradient image
    cv::Mat grad2x(im2.size(), CV_32F);
    cv::Mat grad2y(im2.size(), CV_32F);
    cv::Mat grad1(im1.size(), CV_32F);

    std::vector<cv::Point> candidates;
    for (int x = 1; x < im2.cols - 1; ++x) {
        for (int y = 1; y < im2.rows - 1; ++y) {
            grad2x.ptr<float>(y)[x] =
                (im2.ptr<uchar>(y)[x + 1] - im2.ptr<uchar>(y)[x - 1]) / 2.0f;
            grad2y.ptr<float>(y)[x] =
                (im2.ptr<uchar>(y + 1)[x] - im2.ptr<uchar>(y - 1)[x]) / 2.0f;
            float dx =
                (im1.ptr<uchar>(y)[x + 1] - im1.ptr<uchar>(y)[x - 1]) / 2.0;
            float dy =
                (im1.ptr<uchar>(y + 1)[x] - im1.ptr<uchar>(y - 1)[x]) / 2.0;
            float grad = std::sqrt(dx * dx + dy * dy);
            grad1.ptr<float>(y)[x] = grad;
            if (grad > 50) {
                candidates.push_back(cv::Point(x, y));
            }
        }
    }

    //* build optimization probleml
    ceres::Problem problem;
    double* param = new double[9];
    Eigen::Map<Sophus::SE3d> T(param);
    T = Sophus::SE3d(Eigen::Quaterniond::Identity(),
                     Eigen::Vector3d(-0.07, -0.06, -0.01));  //! T21
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

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.minimizer_progress_to_stdout = true;
    options.linear_solver_type = ceres::DENSE_QR;
    options.num_threads = 1;
    options.update_state_every_iteration = true;
    options.max_num_iterations = 100;
    ceres::Solve(options, &problem, &summary);
    // options.linear_solver_type = ceres::DENSE_SCHUR;

    std::cout << summary.FullReport() << "\n";
    //* visualize the result
    for (size_t i = 0; i < 9; ++i) {
        LOG(ERROR) << "param" << i << " : " << param[i];
    }
    Eigen::Map<Sophus::SE3d> T_x(param);
    for (const auto& p : candidates) {
        if (depth1.at<ushort>(p) == 0) continue;
        Eigen::Vector3d p_3 =
            project2Dto3D(p.x, p.y, depth1.at<ushort>(p), fx, fy, cx, cy, 1000);
        Eigen::Vector3d p_x = T_x * p_3;
        cv::Point p_2 = project3Dto2D(p_x(0), p_x(1), p_x(2), fx, fy, cx, cy);
        cv::circle(im2_clone, p_2, 1, cv::Scalar(0, 255, 0));
    }
    cv::imshow("im2_clone ", im2_clone);
    cv::waitKey(0);

    return 0;
}