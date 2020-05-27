#include "cam_data.h"
#include "config.h"
#include "image_pyramid.h"
#include <eigen3/Eigen/Core>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

using namespace mpl;
int main() {
    //! test config
    Config& config = Config::getInstance();
    std::string config_path = "/home/zhaoran/thesis_ws/mpl/project/config.yaml";
    config.readYamlFile(config_path);

    //! test cam data
    std::string calib_path = "/home/zhaoran/thesis_ws/mpl/project/camera.yaml";

    CamData& cam = CamData::getInstance();
    cam.readYamlFile(calib_path);

    //! test image pyramid
    std::string im_path = "/home/zhaoran/thesis_ws/mpl/test_data/000080.png";
    cv::Mat im = cv::imread(im_path);
    cv::cvtColor(im, im, CV_BGR2GRAY);

    ImagePyramid pyramid(im.data);

    for (int lvl = 0; lvl < cam.lvls; ++lvl) {
        //* color image test
        std::string im_name = "im_at_lvl" + lvl;
        // color data
        ImagePyramid::uchar_ptr row_color_data = pyramid.data(lvl);
        cv::imshow(im_name, cv::Mat(im.rows >> lvl, im.cols >> lvl, CV_8U,
                                    row_color_data.get()));
        cv::waitKey(0);
        // access pxiel value at (u,v) on lvl l

        //* dx image test
        ImagePyramid::float_ptr dx_ptr = pyramid.dx(lvl);
        std::string dx_name = "dx_at_lvl" + lvl;
        cv::imshow(dx_name, cv::Mat(im.rows >> lvl, im.cols >> lvl, CV_32F,
                                    dx_ptr.get()));
        cv::waitKey(0);
        //* dy image test
        auto dy_ptr = pyramid.dy(lvl);
        std::string dy_name = "dy_at_lvl" + lvl;
        cv::imshow(dy_name, cv::Mat(im.rows >> lvl, im.cols >> lvl, CV_32F,
                                    dy_ptr.get()));
        cv::waitKey(0);
        //*  mag test
        auto mag2_ptr = pyramid.mag2(lvl);
        std::string mag2_name = "mag2_at_lvl" + lvl;
        cv::imshow(mag2_name, cv::Mat(im.rows >> lvl, im.cols >> lvl, CV_32F,
                                      mag2_ptr.get()));
        cv::waitKey(0);
    }
}