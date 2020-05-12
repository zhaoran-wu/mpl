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

    std::string im_path = "/home/zhaoran/thesis_ws/mpl/test_data/1.photo.png";
    cv::Mat im = cv::imread(im_path);
    cv::Mat im_clone = im.clone();
    cv::cvtColor(im, im, CV_BGR2GRAY);

    std::string im2_path = "/home/zhaoran/thesis_ws/mpl/test_data/2.photo.png";
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

    cv::Mat depth1 = cv::imread(im_path + "depth/1.png");
    int lvls = config.PYRAMID_LVLS;
    std::vector<cv::Mat> depth_pyramid(lvls);

    // generate image pyramid
    for (int i = 0; i < lvls; ++i) {
        if (i == 0) {
            depth_pyramid[i] = depth1;
        }
        int w_lvl = cam.width[i];
        int h_lvl = cam.height[i];
        for (int x = 0; x < w_lvl; ++x) {
            for (int y = 0; y < h_lvl; ++y) {
                depth_pyramid[i].at<ushort>(y, x) = halfSampling(
                    x, y, cam.width[i - 1], depth_pyramid[i - 1].data);
            }
        }
    }
    // generate pointcloud pyramid
    PointCloudPyramid::ptr pcp(new PointCloudPyramid);
    for (int lvl = 0; lvl < lvls; ++lvl) {
        for (int i = 0; i < candidates.size(); ++i) {
            //! for test we just ignore somce point at each sub lvl
            if (i % lvl != 0) {
                continue;
            }
            if (depth_pyramid[lvl].at<ushort>(candidates[i][1],
                                              candidates[i][0]) == 0) {
                continue;
            }
            Eigen::Vector2f point(
                (float)candidates[i](0) / (i + 1) + 1.f / (i + 1),
                (float)candidates[i](1) / (i + 1) + 1.f / (i + 1));
            Eigen::Vector3f p3d =
                ref->unproject(point,
                               depth_pyramid[lvl].at<ushort>(candidates[i](1),
                                                             candidates[i](0)),
                               lvl);
            // todo make it better
            (*pcp)[lvl].push_back(
                Voxel(p3d, ref->getImagePyramid()->operator()(
                               lvl, candidates[i](0), candidates[i](0))));
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
        cv::resize(im2_clone, tmp, cv::Size(), 1.0f / (i + 1));
        to_track_pyramid.push_back(tmp);
    }

    for (int lvl = 0; lvl < lvls; ++lvl) {
        for (auto& voxel : (*pcp)[lvl]) {
            Eigen::Vector2f projection = ref->project(voxel.position, lvl);
            cv::circle(to_track_pyramid[lvl],
                       cv::Point(projection(0), projection(1)), 1,
                       cv::Scalar(0, 0, 255), 1, CV_FILLED);
        }
        cv::imshow("before optimization", to_track_pyramid[lvl]);
        cv::waitKey(0);
    }

    Sophus::SE3f T_curr_ref(Eigen::Quaternionf::Identity(),
                            Eigen::Vector3f::Zero());

    AffineLight alight_curr_ref(0, 0);

    bool is_success = tracker.tracking(to_track, T_curr_ref, alight_curr_ref);

    for (int lvl = 0; lvl < lvls; ++lvl) {
        for (auto& voxel : (*pcp)[lvl]) {
            Eigen::Vector2f projection =
                ref->project(T_curr_ref.rotationMatrix() * voxel.position +
                                 T_curr_ref.translation(),
                             lvl);
            cv::circle(to_track_pyramid[lvl],
                       cv::Point(projection(0), projection(1)), 1,
                       cv::Scalar(255, 0, 0), 1, CV_FILLED);
        }
        cv::waitKey(0);
        cv::imshow("ater optimization", to_track_pyramid[lvl]);
    }
}