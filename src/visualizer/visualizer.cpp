#include "visualizer.h"
#include "cam_data.h"
#include "frame.h"
#include "point_cloud_pyramid.h"
#include <algorithm>
#include <opencv2/imgproc.hpp>

namespace mpl {

void Visualizer::init() {
    // img data
    cam_data = &CamData::getInstance();
    config = &Config::getInstance();
    img_cols = cam_data->width[scale_factor];
    img_rows = cam_data->height[scale_factor];

    W = 1.5 * (cam_data->width[0] + UI_W);
    H = cam_data->height[0] * 3;

    pangolin::CreateWindowAndBind("MPL: Visualizer", W, H);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // add panel
    pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_W));

    // add Camera Render Object (for view / scene browsing)
    cam_3d = pangolin::OpenGlRenderState(pangolin::ProjectionMatrix(W, H, 600, 600, W / 2.0, H / 2.0, 0.1, 1000.0),
                                         pangolin::ModelViewLookAt(0, -1, -2, 0, 0, 0, pangolin::AxisNegY));

    // add 3D view(main)
    view_3d = pangolin::CreateDisplay()
                  .SetBounds(0.25, 1.0, pangolin::Attach::Pix(UI_W), 1.0, -W / H)
                  .SetHandler(new pangolin::Handler3D(cam_3d));
}
void Visualizer::run() {
    init();

    // menu
    pangolin::Var<bool> menuShowCurrentFrame("menu.Show Current Frame", true, true);
    pangolin::Var<bool> menuDebugDistanceMap("menu.Show Distance Map", false, true);

    // add img view and set texture
    pangolin::View& view_curr_frame = pangolin::Display("curr frame").SetAspect(img_cols / img_rows);
    pangolin::View& view_key_frame_depth = pangolin::Display("key frame depth").SetAspect(img_cols / img_rows);

    pangolin::GlTexture tex_view_curr_frame(img_cols, img_rows, GL_RGBA8, false, 0, GL_RGBA, GL_UNSIGNED_BYTE);
    pangolin::GlTexture tex_view_key_frame_depth(img_cols, img_rows, GL_RGBA, false, 0, GL_RGBA, GL_UNSIGNED_BYTE);

    pangolin::CreateDisplay()
        .SetBounds(0.0, 0.25, pangolin::Attach::Pix(UI_W), 1.0)
        .SetLayout(pangolin::LayoutEqual)
        .AddDisplay(view_curr_frame)
        .AddDisplay(view_key_frame_depth);

    pangolin::OpenGlMatrix Twc;
    Twc.SetIdentity();

    while (true) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // display 3d
        view_3d.Activate(cam_3d);
        glColor4f(1.0, 1.0, 1.0, 1.0f);
        pangolin::glDrawColouredCube();

        // display  curr frame and key frame depth
        if (curr_frame_changed) {  // double check trick to save time
            std::lock_guard<std::mutex> lg_curr_frame(curr_frame_mutex);
            if (curr_frame_changed) {
                tex_view_curr_frame.Upload(curr_frame_img.data, GL_BGRA, GL_UNSIGNED_BYTE);
                curr_frame_changed = false;
            }
        }

        if (key_frame_depth_changed) {  // double check trick to save time
            std::lock_guard<std::mutex> lg_key_frame_depth(key_frame_depth_mutex);
            if (key_frame_depth_changed) {
                tex_view_key_frame_depth.Upload(key_frame_depth.data, GL_BGRA, GL_UNSIGNED_BYTE);
                key_frame_depth_changed = false;
            }
        }

        view_curr_frame.Activate();
        glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
        tex_view_curr_frame.RenderToViewportFlipY();

        view_key_frame_depth.Activate();
        glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
        tex_view_key_frame_depth.RenderToViewportFlipY();

        pangolin::FinishFrame();

        // other images showed by opencv
        if (menuDebugDistanceMap && !config->DEBUG_DISTANCE_MAP) {
            config->debug_distance_map_mutex.lock();
            if (menuDebugDistanceMap) {
                config->DEBUG_DISTANCE_MAP = true;
                config->debug_distance_map_mutex.unlock();
            }
        } else if (!menuDebugDistanceMap && config->DEBUG_DISTANCE_MAP) {
            config->debug_distance_map_mutex.lock();
            if (menuDebugDistanceMap) {
                config->DEBUG_DISTANCE_MAP = false;
                config->debug_distance_map_mutex.unlock();
            }
        }
    }
}

void Visualizer::publish_curr_frame_img(cv::Mat img) {
    std::lock_guard<std::mutex> lg_curr_frame(curr_frame_mutex);
    this->curr_frame_img = img;
    this->curr_frame_changed = true;
}

void Visualizer::publish_key_frame_depth(cv::Mat img) {
    std::lock_guard<std::mutex> lg_key_frame_depth(key_frame_depth_mutex);
    this->key_frame_depth = img;
    this->key_frame_depth_changed = true;
}
void Visualizer::draw_and_publish_curr_frame(std::shared_ptr<PointCloudPyramid> pcp, cv::Mat img,
                                             const Sophus::SE3f& T_curr_KF) {
    // compute statistic
    int point_size = pcp->operator[](0).size();
    std::vector<float> valid_depth_vec;
    valid_depth_vec.reserve(point_size);

    for (const auto& point : pcp->operator[](0)) {
        if (!point.visible_for_newst_frame) continue;
        if (isfinite(point.depth_in_newst_frame) && point.depth_in_newst_frame > 1e-10) {
            valid_depth_vec.push_back(point.depth_in_newst_frame);
        }
    }

    const float lo_percent = 0.01f;
    const float hi_percent = 0.99f;

    float lo, hi;
    std::sort(valid_depth_vec.begin(), valid_depth_vec.end());
    lo = valid_depth_vec[lo_percent * valid_depth_vec.size()];
    hi = valid_depth_vec[hi_percent * valid_depth_vec.size()];

    cv::Mat result = img.clone();
    cv::Mat depth_map = cv::Mat(result.size(), CV_8UC1, cv::Scalar(0));

    for (auto& point : pcp->operator[](0)) {
        if (!point.visible_for_newst_frame) continue;
        Eigen::Vector2f hit_pixel = point.hit_pixel_in_newst_frame;

        float depth = point.depth_in_newst_frame;
        if (!isfinite(depth) || depth < 1e-10) {
            continue;
        } else if (depth < lo) {
            depth = lo;
        } else if (depth > hi) {
            depth = hi;
        }

        uchar wrapped_depth = 255 * std::pow((depth - lo) / (hi - lo), 0.9f);

        cv::circle(depth_map, cv::Point2f(hit_pixel(0), hit_pixel(1)), 1, cv::Scalar(wrapped_depth), 2);
    }
    cv::Mat mask = depth_map.clone();
    cv::applyColorMap(depth_map, depth_map, cv::COLORMAP_JET);
    depth_map.copyTo(result, mask);

    cv::cvtColor(result, result, cv::COLOR_BGR2RGBA);
    publish_curr_frame_img(resize(result));
}

void Visualizer::draw_and_publish_key_frame_depth(cv::Mat depth_im) {
    // compute statistic
    std::vector<ushort> valid_depth_vec(depth_im.rows * depth_im.cols);
    for (int r = 0; r < depth_im.rows; r++) {
        for (int c = 0; c < depth_im.cols; c++) {
            const ushort depth = depth_im.at<ushort>(r, c);
            if (isfinite(depth) && depth > 1e-10) {
                valid_depth_vec.push_back(depth);
            }
        }
    }

    const float lo_percent = 0.01f;
    const float hi_percent = 0.99f;

    float lo, hi;
    std::sort(valid_depth_vec.begin(), valid_depth_vec.end());
    lo = valid_depth_vec[lo_percent * valid_depth_vec.size()];
    hi = valid_depth_vec[hi_percent * valid_depth_vec.size()];

    cv::Mat wrapped_depth_im = cv::Mat::zeros(depth_im.size(), CV_8UC1);
    // set wrapped depth value
    for (int r = 0; r < depth_im.rows; ++r) {
        for (int c = 0; c < depth_im.cols; ++c) {
            ushort depth = depth_im.at<ushort>(r, c);
            if (!isfinite(depth) || depth < 1e-10) {
                continue;
            } else if (depth < lo) {
                depth = lo;
            } else if (depth > hi) {
                depth = hi;
            }
            wrapped_depth_im.at<uchar>(r, c) = 255 * std::pow((depth - lo) / (hi - lo), 0.9f);
        }
    }

    // jet map
    cv::Mat mask = wrapped_depth_im.clone();
    cv::applyColorMap(wrapped_depth_im, wrapped_depth_im, cv::COLORMAP_JET);

    cv::Mat result = cv::Mat::zeros(depth_im.size(), CV_8UC3);
    wrapped_depth_im.copyTo(result, mask);

    cv::cvtColor(result, result, cv::COLOR_BGR2RGBA);

    publish_key_frame_depth(resize(result));
}

inline cv::Mat Visualizer::resize(cv::Mat im) const {
    cv::resize(im, im, cv::Size(im.cols >> scale_factor, im.rows >> scale_factor));
    return im;
}

}  // namespace mpl
