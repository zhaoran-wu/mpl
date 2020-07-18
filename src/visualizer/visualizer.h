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
    bool should_stop = false;

    cv::Mat curr_frame_img;
    cv::Mat key_frame_img;
    cv::Mat key_frame_depth;

    bool curr_frame_changed = false;
    bool key_frame_changed = false;
    bool key_frame_depth_changed = false;

    float W = 1024;
    float H = 768;
    float UI_W = 175;
};

}  // namespace mpl