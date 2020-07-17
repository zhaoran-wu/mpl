#include <opencv2/highgui.hpp>
#include <string>

/**
 * @brief : 1. visualize scene(e.g  point cloud, trajectory, ba window)
 *          2. show image (e.g synetic_im, drawed image, frame image)
 *          3. set param with panel
 *
 */
class Visualizer {
    void run();
    void close();

    void add_image_to_show(const std::string name, cv::Mat image);
};