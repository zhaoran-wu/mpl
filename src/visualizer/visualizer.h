#pragma once
#include <opencv2/highgui.hpp>
#include <pangolin/pangolin.h>
#include <string>

/**
 * @brief : 1. visualize scene(e.g  point cloud, trajectory, ba window)
 *          2. show image (e.g synetic_im, drawed image, frame image)
 *          3. set param with panel
 *
 */
namespace mpl {

class Visualizer {
   public:
    Visualizer() = default;
    void run();
    void close();

    void publish_curr_frame_img(cv::Mat img);
    void publish_key_frame_img(cv::Mat img);
    void publish_key_frame_depth(cv::Mat img);

    void add_image_to_show(const std::string name, cv::Mat image);

   private:
    void init();
    bool should_stop = false;

    // curr frame
    std::mutex curr_frame_mutex;
    cv::Mat curr_frame_img;
    bool curr_frame_changed = false;

    // key frame depth
    std::mutex key_frame_depth_mutex;
    cv::Mat key_frame_depth;
    bool key_frame_depth_changed = false;

    // opengl object
    pangolin::View view_3d;
    pangolin::OpenGlRenderState cam_3d;

    // general settings of window
    const float W = 1480.0f;
    const float H = 960.0f;
    const float UI_W = 175.0f;

    float img_cols;
    float img_rows;
};

}  // namespace mpl