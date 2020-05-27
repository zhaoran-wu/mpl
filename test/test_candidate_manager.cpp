#include "cam_data.h"
#include "candidate_manager.h"
#include "config.h"
#include "sliding_window.h"
#include "synetic_image.h"
#include "tracker.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace mpl;
int main() {
    std::string project_path = "/home/zhaoran/thesis_ws/mpl/project/";
    // read yaml
    Config& config = Config::getInstance();
    std::string config_path = project_path + "config.yaml";
    config.readYamlFile(config_path);

    CamData& cam = CamData::getInstance();
    std::string calib_path = project_path + "camera.yaml";
    cam.readYamlFile(calib_path);

    // read mesh model, texture,initial synetic image
    SyneticImage synetic_image(project_path);

    std::string pose_file = project_path + "ground_truth.txt";
    trajectory_io::Trajectory pose;
    pose.read(pose_file, trajectory_io::Trajectory::FORMAT_MAT);

    // read 5 image date for test
    std::string data_path = "/home/zhaoran/dataset/KITTI/sequences/00/image_0/";
    std::vector<cv::Mat> img_vec;
    std::vector<cv::Mat> img_draw_vec;
    for (int i = 0; i < 5; ++i) {
        int im_id = 80 + i;
        string im_name = (im_id < 100) ? "0000" + to_string(im_id) + ".png"
                                       : "000" + to_string(im_id) + ".png";
        cv::Mat im_tmp = cv::imread(data_path + im_name);
        assert(im_tmp.isContinuous());

        img_draw_vec.push_back(im_tmp.clone());
        cv::cvtColor(im_tmp, im_tmp, CV_BGR2GRAY);
        assert(im_tmp.channels() == 1);
        img_vec.push_back(im_tmp);
    }
    //! we use 4 image to update the candidate depth on the first frame, assume
    //! first frame is KF, and the last 4 are not
    //! we initialize the depth on 1st KF using the synetic depth

    // system begin

    Eigen::Isometry3f T_w_c0 = pose.atIndex(0);
    Eigen::Isometry3f render_pose = T_w_c0;  // T_w_c0 * T_c0_lastKF;
    std::vector<cv::Mat> syn_im_vec = synetic_image.renderingAt(render_pose);
    cv::Mat depth_im = syn_im_vec[1];

    PixelSelector selector;
    std::vector<Eigen::Vector3i> candidates;
    Frame::ptr key_frame = Frame::create(img_vec[0]);
    selector.select(key_frame->getImagePyramid(), candidates);

    // generate depth map vec
    int lvls = config.PYRAMID_LVLS;
    std::vector<cv::Mat> depth_pyramid;
    for (int i = 0; i < lvls; ++i) {
        if (i == 0) {
            depth_pyramid.push_back(depth_im);
            cv::imshow("depth", depth_im);
            cv::waitKey(0);
            continue;
        }
        cv::Mat tmp(cam.height[i], cam.width[i], CV_16U);
        cv::resize(depth_im, tmp, cv::Size(cam.width[i], cam.height[i]));
        depth_pyramid.push_back(tmp);
    }

    // generate pointcloud pyramid
    cv::Mat key_frame_vis = img_draw_vec[0];
    PointCloudPyramid::ptr pcp(new PointCloudPyramid);
    for (int lvl = 0; lvl < lvls; ++lvl) {
        for (size_t i = 0; i < candidates.size(); ++i) {
            //! for test we just ignore somce point at each sub lvl
            if (lvl != 0 && i % lvl != 0) {
                continue;
            }
            if (depth_pyramid[lvl].at<ushort>(
                    candidates[i](1) / (float)pow(2, lvl),
                    candidates[i](0) / (float)pow(2, lvl)) == 0 ||
                depth_pyramid[lvl].at<ushort>(
                    candidates[i](1) / (float)pow(2, lvl),
                    candidates[i](0) / (float)pow(2, lvl)) ==
                    std::numeric_limits<ushort>::max()) {
                continue;
            }
            cv::circle(key_frame_vis,
                       cv::Point2f(candidates[i](0), candidates[i](1)), 1,
                       cv::Scalar(0, 255, 0), 2);

            Eigen::Vector2i point(candidates[i](0) / (float)pow(2, lvl),
                                  candidates[i](1) / (float)pow(2, lvl));
            Eigen::Vector3f p3d = key_frame->unproject(
                point,
                depth_pyramid[lvl].at<ushort>(
                    candidates[i](1) / float(pow(2, lvl)),
                    candidates[i](0) / float(pow(2, lvl))) /
                    1000.f,
                lvl);
            std::cerr << " depth " << p3d(2) << '\n';
            // todo make it better
            int intensity = key_frame->getImagePyramid()->operator()(
                lvl, candidates[i](0) >> (lvl), candidates[i](1) >> (lvl));
            (*pcp)[lvl].push_back(Voxel(p3d, intensity));
        }
    }

    Tracker tracker;
    tracker.set_tracking_ref(key_frame, pcp);

    Sophus::SE3f in_out_T_curr_KF = Sophus::SE3f(Eigen::Quaternionf::Identity(),
                                                 Eigen::Vector3f(0, 0, -0.5));
    AffineLight in_out_aff_light_curr_KF(0, 0);

    cv::imshow("KF", key_frame_vis);
    for (int i = 1; i < 5; ++i) {
        Frame::ptr curr_frame = Frame::create(img_vec[i]);

        Sophus::SE3f old_T_curr_KF = in_out_T_curr_KF;

        tracker.tracking(curr_frame, in_out_T_curr_KF,
                         in_out_aff_light_curr_KF);

        cv::Mat im_to_vis = img_draw_vec[i];
        for (auto& pcd : pcp->operator[](0)) {
            Eigen::Vector3f point = in_out_T_curr_KF * pcd.position;
            Eigen::Vector3f point_no_op = old_T_curr_KF * pcd.position;
            Eigen::Vector2f hit_pixel = curr_frame->project(point);
            Eigen::Vector2f hit_pixel_no_op = curr_frame->project(point_no_op);
            cv::circle(im_to_vis,
                       cv::Point2f(hit_pixel_no_op(0), hit_pixel_no_op(1)), 1,
                       cv::Scalar(0, 0, 255), 1);
            cv::circle(im_to_vis, cv::Point2f(hit_pixel(0), hit_pixel(1)), 1,
                       cv::Scalar(0, 255, 0), 1);
        }

        cv::imshow("curr_frame", im_to_vis);
        cv::waitKey(0);
    }
}