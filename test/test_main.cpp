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
    Eigen::Isometry3f T_w_c0 = pose.atIndex(0);

    // read 100 image date for test
    std::string data_path = "/home/zhaoran/dataset/KITTI/sequences/00/image_0/";

    std::vector<cv::Mat> img_vec;
    std::vector<cv::Mat> img_draw_vec;
    for (int i = 0; i < 100; ++i) {
        int im_id = 80 + i;
        string im_name = (im_id < 100) ? "0000" + to_string(im_id) + ".png" : "000" + to_string(im_id) + ".png";
        cv::Mat im_tmp = cv::imread(data_path + im_name);
        assert(im_tmp.isContinuous());

        img_draw_vec.push_back(im_tmp.clone());
        cv::cvtColor(im_tmp, im_tmp, CV_BGR2GRAY);
        assert(im_tmp.channels() == 1);
        img_vec.push_back(im_tmp);
    }

    Tracker tracker;
    //    SlidingWindow sw;

    // system begin

    Eigen::Isometry3f T_c0_curr = Eigen::Isometry3f::Identity();  // accumulate tranform relative to first cam frame
    for (int i = 0; i < 100; ++i) {
        Frame::ptr curr_frame = Frame::create(img_vec[i]);

        /*         if (sw.empty()) {
                    sw.add_tracked_frame(curr_frame, Sophus::SE3f(Eigen::Quaternionf::Identity(),
           Eigen::Vector3f::Zero()), AffineLight(0, 0));
                } */

        Eigen::Isometry3f render_pose = T_w_c0 * T_c0_curr;
        std::vector<cv::Mat> syn_im_vec = synetic_image.renderingAt(render_pose);
        cv::Mat photo_im = syn_im_vec[0];
        cv::Mat depth_im = syn_im_vec[1];
        cv::Mat normal_im = syn_im_vec[2];
        cv::imshow("photo_im", photo_im);
        cv::imshow("depth_im", depth_im);
        cv::imshow("norma_im", normal_im);
        cv::waitKey(0);

        // todo init point cloud ref from (rendering image + semi-dense map)
        /*         std::vector<Candidate> semi_dense_depth = sw.get_depthmap();

                PointCloudPyramid::ptr pcp = PointCloudPyramid::create(depth_im, semi_dense_depth);

                tracker.set_tracking_ref(pcp);
                Sophus::SE3f T_curr_lastKf = Sophus::SE3f(Eigen::Quaternionf::Identity(), Eigen::Vector3f::Zero());
                AffineLight aff_light_curr_lastKF(0, 0);

                tracker.tracking(curr_frame, T_curr_lastKf, aff_light_curr_lastKF);

                sw.add_tracked_frame(curr_frame, T_curr_lastKf, aff_light_curr_lastKF); */
    }
}