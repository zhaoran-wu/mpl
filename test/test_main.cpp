#include "cam_data.h"
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
    std::string calib_path = project_path + "camera_tum.yaml";
    cam.readYamlFile(calib_path);

    // read mesh model, texture,initial synetic image
    mpl::SyneticImage synetic_image(project_path);

    std::string pose_file = project_path + "ground_truth.txt";
    trajectory_io::Trajectory pose;
    pose.read(pose_file, trajectory_io::Trajectory::FORMAT_MAT);

    // read 100 image date for test
    std::string data_path = "/home/zhaoran/dataset/KITTI/sequences/00/image_0/";
    std::vector<cv::Mat> img_vec;
    std::vector<cv::Mat> img_draw_vec;
    for (int i = 0; i < 100; ++i) {
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

    Tracker tracker;
    SlidingWindow sw;
    Eigen::Isometry3f T_w_c0 = pose.atIndex(0);
    // system begin

    for (int i = 0; i < 100; ++i) {
        Frame::ptr curr_frame = Frame::create(img_vec[i]);

        if (sw.empty()) {
            curr_frame->set_pose(
                Sophus::SE3f(T_w_c0.rotation(), T_w_c0.translation()));
            curr_frame->set_aff_light(0, 0);
            sw.fix_origin(curr_frame);
        }

        Frame::ptr lastKF = sw.get_lastKF();
        Eigen::Isometry3f T_w_lastKF = lastKF->get_pose<Eigen::Isometry3f>();

        Eigen::Isometry3f render_pose = T_w_lastKF;  // T_w_c0 * T_c0_lastKF;
        // todo how to give a global constrain? fix the global pose of the first
        // frame?
        std::vector<cv::Mat> syn_im_vec =
            synetic_image.renderingAt(render_pose);
        cv::Mat depth_im = syn_im_vec[1];
        // todo use depth_im to initilize the search area of candidate

        // todo init point cloud ref from (rendering image + semi-dense map)
        std::vector<Candidate> semi_dense_depth = sw.get_depthmap();

        PointCloudPyramid::ptr pcp =
            PointCloudPyramid::create(depth_im, semi_dense_depth);

        tracker.set_tracking_ref(lastKF, pcp);

        // todo get inital value from KF history
        Sophus::SE3f in_out_T_curr_lastKF = Sophus::SE3f(
            Eigen::Quaternionf::Identity(), Eigen::Vector3f::Zero());
        AffineLight in_out_aff_light_curr_lastKF(0, 0);

        tracker.tracking(curr_frame, in_out_T_curr_lastKF,
                         in_out_aff_light_curr_lastKF);

        // visulization tracking result
        cv::Mat im_to_vis = img_draw_vec[i];
        for (auto& pcd : pcp->operator[](0)) {
            Eigen::Vector3f point = in_out_T_curr_lastKF * pcd.position;
            Eigen::Vector2f hit_pixel = curr_frame->project(point);

            // todo radius is correpond to covariance, color is correspond to
            // depth,
            cv::circle(img_draw_vec, cv::Point2f(hit_pixel(0), hit_pixel(1)), 2,
                       cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("curr_frame", im_to_vis);
        cv::waitKey(0);
        auto& T_curr_lastKF = in_out_T_curr_lastKF;
        auto& aff_light_curr_lastKF = in_out_aff_light_curr_lastKF;
        curr_frame->set_state(T_curr_lastKF, aff_light_curr_lastKF);
        sw.add_tracked_frame(curr_frame, syn_im_vec);

        sw.optimize_window();
    }
}