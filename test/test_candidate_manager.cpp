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
    for (int i = 0; i < 8; ++i) {
        int im_id = 80 + i;
        string im_name = (im_id < 100) ? "0000" + to_string(im_id) + ".png" : "000" + to_string(im_id) + ".png";
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

    Frame::ptr key_frame = Frame::create(img_vec[0]);

    CandidateManager cm;
    cm.select_candidate(key_frame, syn_im_vec[1]);
    std::vector<Candidate>& candidates = cm.get_candidate(key_frame);

    // generate depth map pyramid
    /*     int lvls = config.PYRAMID_LVLS;
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
        } */

    // generate pointcloud pyramid
    /* cv::Mat key_frame_vis = img_draw_vec[0];
    PointCloudPyramid::ptr pcp(new PointCloudPyramid);
    for (int lvl = 0; lvl < lvls; ++lvl) {
        for (size_t i = 0; i < candidates.size(); ++i) {
            //! for test we just ignore somce point at each sub lvl
            if (lvl != 0 && i % lvl != 0 || !candidates[i].is_depth_safe) {
                continue;
            }
            if (depth_pyramid[lvl].at<ushort>(
                    candidates[i].v / (float)pow(2, lvl),
                    candidates[i].u / (float)pow(2, lvl)) == 0 ||
                depth_pyramid[lvl].at<ushort>(
                    candidates[i].v / (float)pow(2, lvl),
                    candidates[i].u / (float)pow(2, lvl)) ==
                    std::numeric_limits<ushort>::max()) {
                continue;
            }
            cv::circle(key_frame_vis,
                       cv::Point2f(candidates[i].u, candidates[i].v), 1,
                       cv::Scalar(0, 255, 0), 2);

            Eigen::Vector2i point(candidates[i].u / (float)pow(2, lvl),
                                  candidates[i].v / (float)pow(2, lvl));
            Eigen::Vector3f p3d =
                key_frame->unproject(point,
                                     depth_pyramid[lvl].at<ushort>(
                                         candidates[i].v / float(pow(2, lvl)),
                                         candidates[i].u / float(pow(2, lvl))) /
                                         1000.f,
                                     lvl);
            std::cerr << " depth " << p3d(2) << '\n';
            // todo make it better
            int intensity = key_frame->get_image_pyramid()->operator()(
                lvl, candidates[i].u >> (lvl), candidates[i].v >> (lvl));
            (*pcp)[lvl].push_back(Voxel(p3d, intensity));
        }
    } */

    PointCloudPyramid::ptr pcp = cm.get_point_cloud_pyramid();
    Tracker tracker;
    tracker.set_tracking_ref(key_frame, pcp);

    Frame::ptr last_frame = key_frame;

    // draw candidates before and after tracking
    for (int i = 1; i < 8; ++i) {
        Frame::ptr curr_frame = Frame::create(img_vec[i]);

        Sophus::SE3f T_last_KF = last_frame->get_T_curr_refKF();
        tracker.tracking(curr_frame);
        Sophus::SE3f T_curr_KF = curr_frame->get_T_curr_refKF();

        cv::Mat im_depth_before_after = img_draw_vec[i].clone();

        // draw before depth update
        /*         for (const auto& can : candidates) {
                    Eigen::Vector2i pixle(can.u, can.v);
                    Eigen::Vector3f P = key_frame->unproject(pixle, 1.0f /
           can.d_inv); Eigen::Vector2f p = curr_frame->project(in_out_T_curr_KF
           * P); cv::circle(im_depth_before_after, cv::Point2f(p(0), p(1)), 3,
                               cv::Scalar(255, 255, 255), 2);
                } */

        cm.update_depth_per_frame(curr_frame);
        auto& candidates_updated = cm.get_candidate(key_frame);
        // draw after depth update
        for (const auto& can : candidates_updated) {
            Eigen::Vector2i pixle(can.u, can.v);

            Eigen::Vector3f P = key_frame->unproject(pixle, can.d_inv);
            // std::cout << "can.inv " << can.d_inv << '\n';
            Eigen::Vector2f p = curr_frame->project(T_curr_KF * P);
            // draw with status
            /*             cv::Scalar color;
                        if (can.status == CandidateStatus::IS_MAP_POINT) {
                            color = cv::Scalar(255, 0, 0);  // blue
                        } else if (can.status ==
               CandidateStatus::NOT_MAP_BUT_CONVERGE) { color = cv::Scalar(0,
               255, 0);  // green } else if (can.status ==
               CandidateStatus::INITIALIZED) { color = cv::Scalar(0, 0, 255); //
               red } else if (can.status == CandidateStatus::OOB) { color =
               cv::Scalar(0, 255, 255);  // yellow } else { color =
               cv::Scalar(255, 0, 255);
                        }

                        cv::circle(im_depth_before_after, cv::Point2f(p(0),
               p(1)), 1, color, 2); */

            // draw with quality

            cv::Scalar color = cv::Scalar(255, 0, 255);
            if (can.status != CandidateStatus::OUTLIER && can.status != CandidateStatus::BAD) {
                cv::circle(im_depth_before_after, cv::Point2f(p(0), p(1)), 1, color, 2);
            }
        }

        // draw before after optimization

        cv::Mat im_to_vis = img_draw_vec[i];
        cv::Mat key_frame_vis = img_draw_vec[0];
        for (auto& pcd : pcp->operator[](0)) {
            Eigen::Vector3f point = T_curr_KF * pcd.position;
            Eigen::Vector3f point_before_optimization = T_last_KF * pcd.position;
            Eigen::Vector2f hit_pixel = curr_frame->project(point);
            Eigen::Vector2f hit_pixel_no_op = curr_frame->project(point_before_optimization);

            Eigen::Vector2f candidate = key_frame->project(pcd.position);

            cv::circle(key_frame_vis, cv::Point2f(candidate(0), candidate(1)), 1, cv::Scalar(0, 0, 255), 2);

            cv::circle(im_to_vis, cv::Point2f(hit_pixel_no_op(0), hit_pixel_no_op(1)), 1, cv::Scalar(0, 0, 255), 2);
            cv::circle(im_to_vis, cv::Point2f(hit_pixel(0), hit_pixel(1)), 1, cv::Scalar(0, 255, 0), 2);
        }
        cv::imshow("KF", key_frame_vis);
        cv::imshow("curr_frame", im_to_vis);
        cv::imshow("depth update", im_depth_before_after);
        cv::waitKey(0);
    }
}