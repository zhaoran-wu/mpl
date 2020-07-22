
#include "../util/tictoc.h"
#include "../util/trajectory.h"
#include "synetic_image.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <string>

int main() {
    std::string project_path = "/home/zhaoran/thesis_ws/mpl/project/";
    mpl::SyneticImage sm(project_path);

    std::string pose_file = project_path + "ground_truth.txt";
    trajectory_io::Trajectory pose;
    pose.read(pose_file, trajectory_io::Trajectory::FORMAT_MAT);
    Eigen::Isometry3f first_pose = pose.atIndex(0);
    Eigen::Isometry3f second_pose = pose.atIndex(1);

    tictoc::tic();

    cv::Mat photo = sm.renderingAt(first_pose, mpl::RenderingMode::PHTOMETRIC);
    // cv::imshow("result",photo);
    cv::imwrite("/home/zhaoran/thesis_ws/mpl/test_data/1.photo.png", photo);
    // cv::waitKey(0);
    cv::Mat depth = sm.renderingAt(first_pose, mpl::RenderingMode::DEPTH);
    // cv::imshow("depth",depth);
    cv::imwrite("/home/zhaoran/thesis_ws/mpl/test_data/1.depth.png", depth);
    // cv::waitKey(0) ;
    cv::Mat normal = sm.renderingAt(first_pose, mpl::RenderingMode::NORMAL);
    cv::imwrite("/home/zhaoran/thesis_ws/mpl/test_data/1.normal.png", normal);

    cv::Mat top_view = sm.renderingAt(first_pose, mpl::RenderingMode::TOP_VIEW);
    cv::imwrite("/home/zhaoran/thesis_ws/mpl/test_data/1.top_view.png", top_view);
    // cv::imshow("normal",normal);
    // cv::waitKey(0);
    // sm.show(first_pose,mpl::RenderingMode::NORMAL);
    cv::Mat photo1 = sm.renderingAt(second_pose, mpl::RenderingMode::PHTOMETRIC);
    // cv::imshow("result",photo1);
    // cv::imwrite("/home/zhaoran/thesis_ws/mpl/test_data/2.photo.png", photo1);
    // cv::waitKey(0);
    cv::Mat depth1 = sm.renderingAt(second_pose, mpl::RenderingMode::DEPTH);
    // cv::imshow("depth",depth1);
    cv::imwrite("/home/zhaoran/thesis_ws/mpl/test_data/2.depth.png", depth1);
    // cv::waitKey(0) ;
    cv::Mat normal1 = sm.renderingAt(second_pose, mpl::RenderingMode::NORMAL);
    cv::imwrite("/home/zhaoran/thesis_ws/mpl/test_data/2.normal.png", normal1);
    // cv::imshow("normal",normal1);
    // cv::waitKey(0);
    auto t = tictoc::toc() / 1000;
    std::cout << "----- use time ----: " << t << " ms" << '\n';
    std::cout << "----- average -----: " << 1000 / (t / 6) << "Hz" << std::endl;

    tictoc::tic();
    std::vector<cv::Mat> result1 = sm.renderingAt(first_pose);
    std::vector<cv::Mat> result2 = sm.renderingAt(second_pose);

    auto t2 = tictoc::toc() / 1000;

    std::cout << "----- use time ----: " << t2 << " ms" << '\n';
    std::cout << "----- average -----: " << 1000 / (t2 / 6) << "Hz" << std::endl;

    cv::Mat photo_im = result1[0];
    cv::Mat depth_im = result1[1];
    cv::Mat normal_im = result1[2];
    cv::imshow("photo_im", photo_im);
    cv::imshow("depth_im", depth_im);
    cv::imshow("norma_im", normal_im);
    cv::waitKey(0);

    return 0;
}
