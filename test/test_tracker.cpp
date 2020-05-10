#include "affine_light.h"
#include "cam_data.h"
#include "config.h"
#include "frame.h"
#include "pixel_selector.h"
#include "sophus/se3.hpp"
#include "tracking_point_cloud.h"
#include <opencv2/opencv.hpp>
using namespace mpl;
int main() {
    Config& config = Config::getInstance();
    std::string config_path = "/home/zhaoran/thesis_ws/mpl/project/config.yaml";
    config.readYamlFile(config_path);

    std::string calib_path = "/home/zhaoran/thesis_ws/mpl/project/camera.yaml";
    CamData& cam = CamData::getInstance();
    cam.readYamlFile(calib_path);

    std::string im_path = "/home/zhaoran/thesis_ws/mpl/test_data/1.photo.png";
    cv::Mat im = cv::imread(im_path);
    cv::Mat im_clone = im.clone();
    cv::cvtColor(im, im, CV_BGR2GRAY);

    std::string im2_path = "/home/zhaoran/thesis_ws/mpl/test_data/2.photo.png";
    cv::Mat im2 = cv::imread(im2_path);
    cv::Mat im2_clone = im2.clone();
    cv::cvtColor(im2, im2, CV_BGR2GRAY);

    Frame::shared_ptr ref(new Frame(im));
    Frame::shared_ptr to_track(new Frame(im2));

    //! here we only simulator the point cloud generate from point manager
    PixelSelector selector;
    Eigen::Vector3i candidates;
    selector.select(ref, candidates);

    std::vector<PointCloud>
        point_cloud;  // Eigen::Vector3f (x,y,z),float inv_d in ref frame

    cv::Mat depth1 = cv::imread(im_path + "depth/1.png");
    for (auto& can : candidates) {
        can(2) =
            (float)depth1.at<ushort>(cv::Point(candidates(0), candidates(1)));
    }

    Tracker tracker;
    tracker->setTrackingRef(
        ref, point_cloud);  // tracer has it's own movement prediction
    // FrontendSystem->movementPredictor->predict(prediction,frontend_history)
    // tracker->setMovementPrediction(pridiction);

    // visualization
    // before tracking

    for (auto& pcd : point_cloud) {
        Eigen::Vector2f projection = ref.project(pcd.position);
        cv::circle(im2_clone, cv::Point(projection(0), projection(1)), 1,
                   cv::Scalar(0, 0, 255), 1, CV_FILLED);
        cv::imshow("before optimization", im2_clone);
        cv::waitKey(0);
    }

    Sophus::SE3d T_ref_curr(Eigen::Quaterniond::Identity(),
                            Eigen::Vector3d::Zero());

    AffineLight alight_ref_curr(0, 0);

    tracker->tracking(to_track, T_ref_curr, alight_ref_curr);

    for (auto& pcd : point_cloud) {
        Eigen::Vector2f projection =
            ref.project_3D_to_2D(T_ref_curr.inverse() * pcd.position);
        cv::circle(im2_clone, cv::Point(projection(0), projection(1)), 1,
                   cv::Scalar(0, 0, 255), 1, CV_FILLED);
        cv::imshow("before optimization", im2_clone);
        cv::waitKey(0);
    }
}