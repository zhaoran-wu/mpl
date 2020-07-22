
#include "PhotometricBA.h"
#include "cam_data.h"
#include "candidate_manager.h"
#include "config.h"
#include "synetic_image.h"
#include <future>
// clang-format off
#include "../visualizer/visualizer.h"
// clang-format on
#include "tracker.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace mpl;

bool is_newframe_KF(PointCloudPyramid::ptr pcp, const Sophus::SE3f& T_KF_curr, float energy) {
    std::cout << "@@@@@@@@ log norm: " << T_KF_curr.log().norm() << "  angle Y : " << abs(T_KF_curr.angleY())
              << "   energy :" << energy << '\n';
    if (abs(T_KF_curr.angleY()) > 0.06 || T_KF_curr.log().norm() > 1.3 || energy > 350.f) return true;
}
/*
template <typename... Args>
using execute_func = void (*f)(Args...);

template <typename... Args>
void execute_accoding_to_config(bool project_config, std::mutex config_mutex, execute_func<Args...> f,
                                const Args&... args) {
    if (project_config) {
        config_mutex.lock();
        if (project_config) {
            config_mutex.unlock();
            f(args);
            continue;
        }
        config_mutex.unlock();
    }
} */

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

    // read 12 image date for test
    std::string data_path = "/home/zhaoran/dataset/KITTI/sequences/00/image_0/";
    std::vector<cv::Mat> img_vec;
    std::vector<cv::Mat> img_draw_vec;
    for (int i = 0; i < 200; ++i) {
        int im_id = 80 + i;
        string im_name = (im_id < 100) ? "0000" + to_string(im_id) + ".png" : "000" + to_string(im_id) + ".png";
        cv::Mat im_tmp = cv::imread(data_path + im_name);
        assert(im_tmp.isContinuous());

        img_draw_vec.push_back(im_tmp.clone());

        cv::cvtColor(im_tmp, im_tmp, CV_BGR2GRAY);
        assert(im_tmp.channels() == 1);
        img_vec.push_back(im_tmp);
    }

    // test :
    // build a window with 2 old KF and a new added frame
    // use very simple strategy : add KF after 2 frames
    // 1. activate candidate of 2 old KF
    // 2. generate point cloud pyramid on newst frame for tracking

    // system begin

    synetic_image.set_start_pose(pose.atIndex(0));
    Sophus::SE3f render_pose = Sophus::SE3f(Eigen::Quaternionf::Identity(), Eigen::Vector3f(0, 0, 0));

    Frame::ptr key_frame = Frame::create(img_vec[0]);
    std::vector<cv::Mat> syn_im_vec_curr = synetic_image.renderingAt(render_pose);
    std::shared_ptr<Visualizer> vis(new Visualizer);
    CandidateManager cm(vis);
    cm.select_candidate(key_frame, syn_im_vec_curr[1]);
    std::vector<Candidate>& candidates = cm.get_candidate(key_frame);

    PointCloudPyramid::ptr pcp = cm.get_point_cloud_pyramid();

    Tracker tracker;
    tracker.set_tracking_ref(key_frame, pcp);

    // draw candidates before and after tracking
    cv::Mat key_frame_vis = img_draw_vec[0];

    PhotometricBA pba;

    for (size_t i = 1; i < img_vec.size(); ++i) {
        Frame::ptr curr_frame = Frame::create(img_vec[i]);

        // if tracking failed(due to e.g over exposure), throw this frame just away
        if (!tracker.tracking(curr_frame)) continue;

        // visualize tracking result
        std::thread th_draw(&Visualizer::draw_and_publish_curr_frame, std::ref(*vis), pcp, img_draw_vec[i],
                            curr_frame->get_pose());
        th_draw.detach();

        // todo : a best way to choose KF
        Sophus::SE3f T_curr_KF = get_src_to_dst_transform(curr_frame->get_ref_frame(), curr_frame);
        bool is_KF = is_newframe_KF(pcp, T_curr_KF, tracker.get_per_pixel_energy());
        if (is_KF) {
            // rendering at new key frame's pose
            render_pose = curr_frame->get_pose();
            syn_im_vec_curr = synetic_image.renderingAt(render_pose);

            // add new frame to the window, and select candidate on it
            cm.select_candidate(curr_frame, syn_im_vec_curr[1]);

            // optimize key frame window
            pba.solve(cm);

            // get and set new tracking reference point cloud after optimization
            pcp = cm.get_point_cloud_pyramid();
            tracker.set_tracking_ref(curr_frame, pcp);

            // visualize: depth map
            std::thread th_draw_depth(&Visualizer::draw_and_publish_key_frame_depth, std::ref(*vis),
                                      syn_im_vec_curr[1]);
            th_draw_depth.detach();

            // visualization
            /*             execute_accoding_to_config(config.DEBUG_KEY_FRAME_SYNETIC_IMAGE_ALIGNMENT,
                                                   config.debug_key_frame_synetci_img_alignment_mutex,
                                                   show_debug_key_frame_synetic_img_alignment, img_draw_vec[i],
               syn_im_vec_curr[0]); */
        }
    }
}