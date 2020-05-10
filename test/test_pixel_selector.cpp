#include "cam_data.h"
#include "config.h"
#include "image_pyramid.h"
#include "pixel_selector.h"
#include <opencv2/opencv.hpp>

using namespace mpl;

int main() {
    //* pre-processing: config , camera param , image pyramid
    Config& config = Config::getInstance();
    std::string config_path = "/home/zhaoran/thesis_ws/mpl/project/config.yaml";
    config.readYamlFile(config_path);

    std::string calib_path = "/home/zhaoran/thesis_ws/mpl/project/camera.yaml";
    CamData& cam = CamData::getInstance();
    cam.readYamlFile(calib_path);

    std::string im_path = "/home/zhaoran/thesis_ws/mpl/test_data/000080.png";
    cv::Mat im = cv::imread(im_path);
    cv::Mat im_clone = im.clone();
    cv::cvtColor(im, im, CV_BGR2GRAY);

    std::shared_ptr<ImagePyramid> pyramid(new ImagePyramid(im.data));

    //! test pixel selector
    PixelSelector selector;
    std::vector<Eigen::Vector3i> candidate;
    int num = selector.select(pyramid, candidate);
    const auto im_ptr = pyramid->data(0);

    for (auto& p : candidate) {
        if (p(2) == 0) {
            cv::circle(im_clone, cv::Point(p(0), p(1)), 1,
                       cv::Scalar(0, 255, 0), 2, CV_FILLED);
            /*         } else if (p(2) == 1) {
                        cv::circle(im_clone, cv::Point(p(0), p(1)), 2,
                                   cv::Scalar(0, 255, 0));
                    } else {
                        cv::circle(im_clone, cv::Point(p(0), p(1)), 2,
                                   cv::Scalar(255, 0, 0)); */
        }
    }

    cv::imshow("im", im_clone);
    cv::waitKey(0);

    std::string im2_path = "/home/zhaoran/thesis_ws/mpl/test_data/homo.png";
    cv::Mat im2 = cv::imread(im2_path);
    cv::Mat im2_clone = im2.clone();
    cv::cvtColor(im2, im2, CV_BGR2GRAY);

    ImagePyramid::ptr pyramid2(new ImagePyramid(im2.data));

    //! test pixel selector
    std::vector<Eigen::Vector3i> candidate2;
    int num_2 = selector.select(pyramid2, candidate2);
    const auto im_ptr2 = pyramid2->data(0);

    for (auto& p : candidate2) {
        if (p(2) == 0) {
            cv::circle(im2_clone, cv::Point(p(0), p(1)), 1,
                       cv::Scalar(0, 255, 0), 2, CV_FILLED);
            /*         } else if (p(2) == 1) {
                        cv::circle(im2_clone, cv::Point(p(0), p(1)), 2,
                                   cv::Scalar(0, 255, 0));
                    } else {
                        cv::circle(im2_clone, cv::Point(p(0), p(1)), 2,
                                   cv::Scalar(255, 0, 0)) */
            ;
        }
    }

    cv::imshow("im2", im2_clone);
    cv::waitKey(0);
}
