#include "../util/utility.h"
#include "affine_light.h"
#include "cam_data.h"
#include "config.h"
#include "frame.h"
#include "image_pyramid.h"
#include "pixel_selector.h"
#include "point_cloud_pyramid.h"
#include "tracking_optimizer.h"
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

Eigen::Vector3d projectTo3d(const cv::Point pixel, const Eigen::Matrix3d K,
                            const cv::Mat depth) {
    Eigen::Vector3d homo_pixel(pixel.x, pixel.y, 1);
    /*   float d = depth.at<uchar>(pixel) * 40 /
                255.0f;  //*100; //! 40 is z_far //255 -> 256 faster */
    float d = depth.at<ushort>(pixel) * 0.001;
    // LOG(INFO) << " depth " << d << " m";
    return d * K.inverse() * homo_pixel;
}
using namespace mpl;
int main(int argc, char** argv) {
    //* prepare data for optimization problem
    Config& config = Config::getInstance();
    std::string config_path = "/home/zhaoran/thesis_ws/mpl/project/config.yaml";
    config.readYamlFile(config_path);

    std::string calib_path =
        "/home/zhaoran/thesis_ws/mpl/project/camera_tum.yaml";
    CamData& cam = CamData::getInstance();
    cam.readYamlFile(calib_path);
    const std::string im_path = "/home/zhaoran/thesis_ws/mpl/test_data/";
    cv::Mat im1 = cv::imread(im_path + "rgb/1.png");
    cv::Mat im1_clone = im1.clone();
    cv::cvtColor(im1, im1, CV_BGR2GRAY);

    cv::Mat im2 = cv::imread(im_path + "rgb/2.png");
    cv::Mat im2_clone = im2.clone();
    cv::cvtColor(im2, im2, CV_BGR2GRAY);

    cv::Mat depth1 = cv::imread(im_path + "depth/1.png");

    Frame::ptr frame(new Frame(im1.data));
    Frame::ptr frame2(new Frame(im2.data));

    PixelSelector selector;
    std::vector<Eigen::Vector3i> candidates_eigen;
    int num = selector.select(frame->getImagePyramid(), candidates_eigen);

    PointCloudPyramid::ptr pcp;
    PointCloud pcl;

    for (const auto& p : candidates_eigen) {
        pcl.push_back(
            Voxel(frame->unproject(p.topRows(2),
                                   depth1.at<ushort>(cv::Point(p(0), p(1)))),
                  im1.at<uchar>(cv::Point(p(0), p(1)))));
    }
    pcp->operator[](0) = pcl;

    //* initial param
    auto T = Sophus::SE3f(Eigen::Quaternionf::Identity(),
                          Eigen::Vector3f(-0.06, -0.04, 0));  //! T21
    AffineLight affL(0, 0);
    //* build the problem
    TrackingOptimizer tra_optimizer;
    tra_optimizer.init(T, affL, pcp, frame2);

    // before
    for (const auto& p_eigen : candidates_eigen) {
        cv::Point p(p_eigen(0), p_eigen(1));
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

    // solve to get the pose
    float residual = tra_optimizer.solve(10);
    T = tra_optimizer.getT();
    affL = tra_optimizer.getAffineLight();
    //* after
    for (const auto& p_eigen : candidates_eigen) {
        cv::Point p(p_eigen(0), p_eigen(1));
        if (depth1.at<ushort>(p) == 0) continue;
        cv::circle(im1_clone, p, 1, cv::Scalar(0, 255, 0), 1, CV_FILLED);
        Eigen::Vector3d p_3 =
            project2Dto3D(p.x, p.y, depth1.at<ushort>(p), fx, fy, cx, cy, 1000);
        Eigen::Vector3f p_x = T * p_3.cast<float>();
        cv::Point2d p_2 = project3Dto2D(p_x(0), p_x(1), p_x(2), fx, fy, cx, cy);
        cv::Point p_draw(p_2.x, p_2.y);
        cv::circle(im2_clone, p_draw, 1, cv::Scalar(0, 0, 255), 1, CV_FILLED);
    }
    cv::imshow("im2_clone ", im2_clone);
    cv::imshow("im1_clone ", im1_clone);
    cv::waitKey(0);

    return 0;
}