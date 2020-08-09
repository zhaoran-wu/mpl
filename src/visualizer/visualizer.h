#pragma once
#include "../util/kitti.h"
#include "point_cloud_pyramid.h"
#include <opencv2/highgui.hpp>
#include <pangolin/pangolin.h>
#include <queue>
#include <sophus/se3.hpp>
#include <string>

/**
 * @brief : 1. visualize scene(e.g  point cloud, trajectory, ba window)
 *          2. show image (e.g synetic_im, tracking result)
 *          3. set param with menu: e.g if show a debug image
 *
 */
namespace mpl {

class PointCloudPyramid;
class CandidateManager;
class CamData;
class Config;
class Frame;
class Visualizer {
    typedef std::pair<Eigen::Vector3f, Eigen::Vector3f> line;

   public:
    Visualizer();
    Visualizer(const Visualizer&) = delete;
    void run();
    void close();

    // called in main thread
    void publish_curr_frame_tracking_info(std::shared_ptr<PointCloudPyramid> pcp, cv::Mat img,
                                          const Sophus::SE3f& T_w_c, const Sophus::SE3f& T_c_kf,
                                          const bool draw_outlier);

    // called in main thread
    void draw_and_publish_key_frame_depth(cv::Mat depth_im);

    // stop main thread
    // this function need to call in the main thread's main loop
    void stop_or_start_according_to_pangolin_menu();
    std::mutex stop_main_thread_mutex;
    bool stop_main_thread = true;

    // must call before remove
    void set_new_sliding_window_data(CandidateManager& cm, const std::shared_ptr<Frame> to_remove_frame);

   private:
    void draw_tracking_point_cloud();
    void draw_curr_frame_cam();

    void draw_camera(const Sophus::SE3f& T_w_c, const int line_width = 5.0f, const float r = 1.0f, const float g = 0.0f,
                     const float b = 0.0f);
    void draw_cameras(const std::vector<Sophus::SE3f>& pose_vec, const float line_width, const float line_r,
                      const float line_g, const float line_b, const float traj_width, const float traj_r,
                      const float traj_g, const float traj_b);
    void draw_point_clouds(const std::vector<Eigen::Vector3f>& point_vec, const float point_size, const float r,
                           const float g, const float b);
    // line are given the pose in local frame if T_w_o is given, or in global frame(T_w_o is identity)
    void draw_lines(const std::vector<line>& line_vec, const float line_width, const float r, const float g,
                    const float b, const Sophus::SE3f& T_w_o = Sophus::SE3f::transZ(0.0f));
    void draw_sliding_window();
    void draw_pose_history();
    void draw_point_cloud_history();

    cv::Mat resize(cv::Mat im) const;
    void set_key_frame_depth(cv::Mat img);
    void set_curr_frame_tracking_info(cv::Mat img, const Sophus::SE3f& T_w_c, const Sophus::SE3f& T_c_kf);
    void init();

    void draw_ground_truth();

    KittiReader ground_truth;

    // curr frame(tracking)
    std::mutex curr_frame_mutex;
    cv::Mat curr_frame_img;
    Sophus::SE3f T_w_c;
    Sophus::SE3f T_c_kf;
    std::vector<Voxel> pcd;
    bool curr_frame_image_changed = false;

    // newst key frame depth
    std::mutex key_frame_depth_mutex;
    cv::Mat key_frame_depth;
    bool key_frame_depth_changed = false;

    // sliding widow
    std::mutex sliding_window_mutex;
    bool is_sliding_window_changed = false;
    std::vector<Sophus::SE3f> T_w_c_sliding_window;
    std::vector<Eigen::Vector3f> point_cloud_siding_window;

    // all key frame/point cloud history(data that will not change )
    std::mutex history_mutex;
    bool is_history_changed = false;
    std::vector<Sophus::SE3f> T_w_c_history;
    std::vector<Eigen::Vector3f> point_cloud_history;

    // opengl object
    pangolin::View view_3d;
    pangolin::OpenGlRenderState cam_3d;

    // general settings of window
    float W = 640.0f;
    float H = 480.0f;
    const float UI_W = 220.0f;

    // data to draw the camera
    Eigen::Vector3f tl;
    Eigen::Vector3f tr;
    Eigen::Vector3f dl;
    Eigen::Vector3f dr;

    // salce : scale the upload image to reduce data size
    const int scale_factor = 0;

    float img_cols;
    float img_rows;

    CamData* cam_data;
    Config* config;

    std::vector<line> camera_to_draw;
};

}  // namespace mpl