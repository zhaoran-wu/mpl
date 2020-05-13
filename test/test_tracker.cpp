#include "affine_light.h"
#include "cam_data.h"
#include "config.h"
#include "frame.h"
#include "pixel_selector.h"
#include "point_cloud_pyramid.h"
#include "sophus/se3.hpp"
#include "tracker.h"
#include <opencv2/opencv.hpp>
using namespace mpl;
int main() {
    Config& config = Config::getInstance();
    std::string config_path = "/home/zhaoran/thesis_ws/mpl/project/config.yaml";
    config.readYamlFile(config_path);

    std::string calib_path =
        "/home/zhaoran/thesis_ws/mpl/project/camera_tum.yaml";
    CamData& cam = CamData::getInstance();
    cam.readYamlFile(calib_path);

    std::string im_path = "/home/zhaoran/thesis_ws/mpl/test_data/rgb/1.png";
    cv::Mat im = cv::imread(im_path);
    cv::Mat im_clone = im.clone();
    cv::cvtColor(im, im, CV_BGR2GRAY);

    std::string im2_path = "/home/zhaoran/thesis_ws/mpl/test_data/rgb/2.png";
    cv::Mat im2 = cv::imread(im2_path);
    cv::Mat im2_clone = im2.clone();
    cv::cvtColor(im2, im2, CV_BGR2GRAY);

    Frame::ptr ref(new Frame(im.data));
    Frame::ptr to_track(new Frame(im2.data));

    //! here we only simulator the point cloud generate from point manager,since
    //! we do not have pixel manager now
    PixelSelector selector;
    std::vector<Eigen::Vector3i> candidates;
    selector.select(ref->getImagePyramid(), candidates);

    cv::Mat depth1 =
        cv::imread("/home/zhaoran/thesis_ws/mpl/test_data/depth/1.png");
    int lvls = config.PYRAMID_LVLS;
    std::vector<cv::Mat> depth_pyramid;

    // generate image pyramid
    for (int i = 0; i < lvls; ++i) {
        if (i == 0) {
            depth_pyramid.push_back(depth1);
            cv::imshow("depth", depth1);
            cv::waitKey(0);
            continue;
        }
        cv::Mat tmp(cam.height[i], cam.width[i], CV_16U);
        cv::resize(depth1, tmp, cv::Size(cam.width[i], cam.height[i]));
        depth_pyramid.push_back(tmp);
    }
    // generate pointcloud pyramid
    PointCloudPyramid::ptr pcp(new PointCloudPyramid);
    for (int lvl = 0; lvl < lvls; ++lvl) {
        for (int i = 0; i < candidates.size(); ++i) {
            //! for test we just ignore somce point at each sub lvl
            if (lvl != 0 && i % lvl != 0) {
                continue;
            }
            if (depth_pyramid[lvl].at<ushort>(candidates[i](1) >> (lvl),
                                              candidates[i](0) >> (lvl)) == 0) {
                continue;
            }
            Eigen::Vector2i point(candidates[i](0) / (float)pow(2, lvl),
                                  candidates[i](1) / (float)pow(2, lvl));
            Eigen::Vector3f p3d = ref->unproject(
                point,
                depth_pyramid[lvl].at<ushort>(candidates[i](1) >> (lvl),
                                              candidates[i](0) >> (lvl)) /
                    1000.f,
                lvl);
            // todo make it better
            int intensity = ref->getImagePyramid()->operator()(
                lvl, candidates[i](0) >> (lvl), candidates[i](1) >> (lvl));
            (*pcp)[lvl].push_back(Voxel(p3d, intensity));
        }
    }

    Tracker tracker;
    tracker.set_tracking_ref(pcp);  // tracer has it's own movement prediction
    // FrontendSystem->movementPredictor->predict(prediction,frontend_history)
    // tracker->setMovementPrediction(pridiction);

    // visualization
    // before tracking
    std::vector<cv::Mat> to_track_pyramid;
    // generate image pyramid
    for (int i = 0; i < lvls; ++i) {
        cv::Mat tmp;
        cv::resize(im2_clone, tmp, cv::Size(cam.width[i], cam.height[i]));
        to_track_pyramid.push_back(tmp);
    }

    /*     for (int lvl = 0; lvl < lvls; ++lvl) {
            for (auto& voxel : (*pcp)[lvl]) {
                Eigen::Vector2f projection = ref->project(voxel.position, lvl);
                cv::circle(to_track_pyramid[lvl],
                           cv::Point(projection(0), projection(1)), 1,
                           cv::Scalar(0, 0, 255), 1, CV_FILLED);
            }
            cv::imshow("before optimization", to_track_pyramid[lvl]);
            cv::waitKey(0);
        } */

    Sophus::SE3f T_curr_ref(Eigen::Quaternionf::Identity(),
                            Eigen::Vector3f::Zero());

    AffineLight alight_curr_ref(0, 0);

    bool is_success = tracker.tracking(to_track, T_curr_ref, alight_curr_ref);

    for (int lvl = lvls - 1; lvl >= 0; --lvl) {
        for (auto& voxel : (*pcp)[lvl]) {
            Eigen::Vector2f projection =
                ref->project(T_curr_ref.rotationMatrix() * voxel.position +
                                 T_curr_ref.translation(),
                             lvl);
            cv::circle(to_track_pyramid[lvl],
                       cv::Point(projection(0), projection(1)), 1,
                       cv::Scalar(0, 255, 0), 1, CV_FILLED);
        }
        cv::imshow("ater optimization", to_track_pyramid[lvl]);
        cv::waitKey(0);
    }
}
