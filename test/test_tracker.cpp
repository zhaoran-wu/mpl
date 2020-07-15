#include "cam_data.h"
#include "candidate_manager.h"
#include "config.h"
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
    std::string calib_path = project_path + "camera_tum.yaml";
    cam.readYamlFile(calib_path);

    // read 5 image date for test
    std::string data_path = "/home/zhaoran/thesis_ws/mpl/test_data/";
    std::vector<cv::Mat> img_vec;
    std::vector<cv::Mat> depth_vec;
    std::vector<cv::Mat> img_draw_vec;
    for (int i = 15; i < 32; ++i) {
        string im_name = "rgb2/" + to_string(i) + ".png";
        string depth_name = "depth2/" + to_string(i) + ".png";
        cv::Mat im_tmp = cv::imread(data_path + im_name);
        cv::Mat depth_tmp = cv::imread(data_path + depth_name);

        assert(im_tmp.isContinuous());
        assert(depth_tmp.isContinuous());

        img_draw_vec.push_back(im_tmp.clone());
        cv::cvtColor(im_tmp, im_tmp, CV_BGR2GRAY);
        assert(im_tmp.channels() == 1);
        img_vec.push_back(im_tmp);
        depth_vec.push_back(depth_tmp);
    }
    //! we use 4 image to update the candidate depth on the first frame, assume
    //! first frame is KF, and the last 4 are not
    //! we initialize the depth on 1st KF using the synetic depth

    // system begin

    Frame::ptr key_frame = Frame::create(img_vec[0]);
    /*     PixelSelector selector;
        std::vector<Eigen::Vector3i> candidates;
        selector.select(key_frame->getImagePyramid(), candidates) */
    ;

    CandidateManager cm;
    cm.select_candidate(key_frame, depth_vec[0]);
    std::vector<Candidate>& candidates = cm.get_candidate(key_frame);

    // generate depth map vec
    cv::Mat depth_im = depth_vec[0];
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

            int u = candidates[i].u / (float)pow(2, lvl);
            int v = candidates[i].v / (float)pow(2, lvl);

            if (u < 0 || u > depth_pyramid[lvl].cols || v < 0 ||
                v > depth_pyramid[lvl].rows) {
                continue;
            }
            ushort depth = depth_pyramid[lvl].at<ushort>(v, u);
            if (depth == 0 || depth == std::numeric_limits<ushort>::max()) {
                continue;
            }

            if (lvl == 0) {
                cv::circle(key_frame_vis, cv::Point2f(u, v), 1,
                           cv::Scalar(0, 255, 0), 2);
            }
            Eigen::Vector2i point(u, v);
            Eigen::Vector3f p3d =
                key_frame->unproject(point, depth / 1000.f, lvl);
            // todo make it better
            int intensity = key_frame->getImagePyramid()->operator()(lvl, u, v);
            (*pcp)[lvl].push_back(Voxel(p3d, intensity));
        }
    }

    Tracker tracker;
    tracker.set_tracking_ref(key_frame, pcp);

    Sophus::SE3f in_out_T_curr_KF =
        Sophus::SE3f(Eigen::Quaternionf::Identity(), Eigen::Vector3f(0, 0, 0));
    AffineLight in_out_aff_light_curr_KF(0, 0);

    cv::imshow("KF", key_frame_vis);
    Frame::ptr last_frame = key_frame;
    for (int i = 1; i < 16; ++i) {
        Frame::ptr curr_frame = Frame::create(img_vec[i]);

        Sophus::SE3f old_T_curr_KF = last_frame->get_T_curr_lastKF();

        in_out_T_curr_KF = curr_frame->get_T_curr_lastKF();

        cv::Mat im_to_vis = img_draw_vec[i];
        for (auto& pcd : pcp->operator[](0)) {
            Eigen::Vector3f point = in_out_T_curr_KF * pcd.position;
            Eigen::Vector3f point_no_op = old_T_curr_KF * pcd.position;
            Eigen::Vector2f hit_pixel = curr_frame->project(point);
            Eigen::Vector2f hit_pixel_no_op = curr_frame->project(point_no_op);
            // cv::circle(im_to_vis,
            //           cv::Point2f(hit_pixel_no_op(0), hit_pixel_no_op(1)), 1,
            //           cv::Scalar(0, 0, 255), 2);
            cv::circle(im_to_vis, cv::Point2f(hit_pixel(0), hit_pixel(1)), 3,
                       cv::Scalar(0, 255, 0), 3);
        }

        cm.update_depth_per_frame(curr_frame);
        auto& cans = cm.get_candidate(key_frame);
        for (const auto& can : cans) {
            Eigen::Vector2i pixle(can.u, can.v);
            Eigen::Vector3f P =
                key_frame->unproject(pixle, 1.0f / can.d_inv_synetic_im);
            Eigen::Vector2f p = curr_frame->project(in_out_T_curr_KF * P);
            cv::circle(im_to_vis, cv::Point2f(p(0), p(1)), 2,
                       cv::Scalar(0, 0, 255), 2);
        }
        cv::imshow("curr_frame", im_to_vis);
        cv::waitKey(0);

        last_frame = curr_frame;
    }
}
