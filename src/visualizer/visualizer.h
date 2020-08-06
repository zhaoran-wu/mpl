#pragma once
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

    void draw_cameras(const std::vector<Sophus::SE3f>& pose_vec);
    void draw_point_clouds(const std::vector<Eigen::Vector3f>& point_vec, const float point_size, const float r,
                           const float g, const float b);
    // line are given the pose in local frame if T_w_o is given, or in global frame(T_w_o is identity)
    void draw_lines(const std::vector<line>& line_vec, const float line_width, const float r, const float g,
                    const float b, const Sophus::SE3f& T_w_o = Sophus::SE3f::transZ(0.0f));
    void draw_sliding_window();
    void draw_all_history();

    void draw_cam(const Sophus::SE3f& T_w_c);
    cv::Mat resize(cv::Mat im) const;
    void set_key_frame_depth(cv::Mat img);
    void set_curr_frame_tracking_info(cv::Mat img, const Sophus::SE3f& T_w_c, const Sophus::SE3f& T_c_kf);
    void init();

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
    std::vector<Sophus::SE3f> T_w_c_sliding_window;
    std::vector<Eigen::Vector3f> point_cloud_siding_window;

    // all key frame/point cloud history(data that will not change )
    std::mutex history_mutex;
    std::queue<Sophus::SE3f> T_w_c_history;
    std::queue<std::vector<Eigen::Vector3f>> point_cloud_history;

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

    const float POINT_SIZE = 3.0f;
};

}  // namespace mpl