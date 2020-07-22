#pragma once
#include <opencv2/highgui.hpp>
#include <pangolin/pangolin.h>
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
class CamData;
class Config;
class Visualizer {
   public:
    Visualizer();
    Visualizer(const Visualizer&) = delete;
    void run();
    void close();

    void draw_and_publish_curr_frame(std::shared_ptr<PointCloudPyramid> pcp, cv::Mat img, const Sophus::SE3f& T_w_c);

    void draw_and_publish_key_frame_depth(cv::Mat depth_im);

   private:
    void draw_cam();
    cv::Mat resize(cv::Mat im) const;
    void publish_key_frame_depth(cv::Mat img);
    void publish_curr_frame_img(cv::Mat img, const Sophus::SE3f& T_w_c);
    void init();
    bool should_stop = false;

    // curr frame
    std::mutex curr_frame_mutex;
    cv::Mat curr_frame_img;
    Sophus::SE3f T_w_c;
    bool curr_frame_image_changed = false;

    // key frame depth
    std::mutex key_frame_depth_mutex;
    cv::Mat key_frame_depth;
    bool key_frame_depth_changed = false;

    // opengl object
    pangolin::View view_3d;
    pangolin::OpenGlRenderState cam_3d;

    // general settings of window
    float W = 640.0f;
    float H = 480.0f;
    const float UI_W = 175.0f;

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
};

}  // namespace mpl