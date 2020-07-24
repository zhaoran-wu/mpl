#include "visualizer.h"
#include "cam_data.h"
#include "frame.h"
#include "point_cloud_pyramid.h"
#include <algorithm>
#include <chrono>
#include <opencv2/imgproc.hpp>
#include <thread>

namespace mpl {
inline void check_and_change_config_according_to_menu(pangolin::Var<bool>& menu_config, bool& project_config,
                                                      std::mutex& project_config_mutex);

Visualizer::Visualizer() {
    std::thread th(&Visualizer::run, std::ref(*this));
    th.detach();
}

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
    cam_3d = pangolin::OpenGlRenderState(pangolin::ProjectionMatrix(W, H, 600, 600, W / 2.0, H / 2.0, 0.01, 1000.0),
                                         pangolin::ModelViewLookAt(0, 10, -10, 0, 0, 0, pangolin::AxisNegY));

    // add 3D view(main)
    view_3d = pangolin::CreateDisplay()
                  .SetBounds(0.25, 1.0, pangolin::Attach::Pix(UI_W), 1.0, -W / H)
                  .SetHandler(new pangolin::Handler3D(cam_3d));

    // data to draw the camera frame
    float scale = 1.0f;
    tl = unproject(cam_data, Eigen::Vector2i(0, 0), 1 / scale);
    tr = unproject(cam_data, Eigen::Vector2i(img_cols - 1, 0), 1 / scale);
    dl = unproject(cam_data, Eigen::Vector2i(0, img_rows - 1), 1 / scale);
    dr = unproject(cam_data, Eigen::Vector2i(img_cols - 1, img_rows - 1), 1 / scale);
}
void Visualizer::run() {
    init();

    // menu
    pangolin::Var<bool> menuStop("menu.Stop", false, true);
    pangolin::Var<bool> menuDebugDistanceMap("menu.Debug Distance Map", false, true);
    pangolin::Var<bool> menuShowKeyFrameSyneticImageAlignment("menu.Debug Image Alignment", false, true);
    pangolin::Var<bool> menuDebugPixelSelection("menu.Debug Pixel Selection", false, true);
    pangolin::Var<bool> menuDebugImagePyramid("menu.Debug Image Pydamid", false, true);
    pangolin::Var<bool> menuDebugCoarseToFineTracking("menu.Debug Pyramid Tracking", false, true);

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

    while (!pangolin::ShouldQuit()) {
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // display 3d
        view_3d.Activate(cam_3d);
        glColor4f(1.0, 1.0, 1.0, 1.0f);
        draw_curr_frame_cam();

        // display  curr frame and key frame depth
        if (curr_frame_image_changed) {  // double check trick to save time
            std::lock_guard<std::mutex> lg_curr_frame(curr_frame_mutex);
            if (curr_frame_image_changed) {
                tex_view_curr_frame.Upload(curr_frame_img.data, GL_BGRA, GL_UNSIGNED_BYTE);
                curr_frame_image_changed = false;
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
        check_and_change_config_according_to_menu(menuStop, this->stop_main_thread, this->stop_main_thread_mutex);

        check_and_change_config_according_to_menu(menuDebugDistanceMap, config->DEBUG_DISTANCE_MAP,
                                                  config->debug_distance_map_mutex);

        check_and_change_config_according_to_menu(menuShowKeyFrameSyneticImageAlignment,
                                                  config->DEBUG_KEY_FRAME_SYNETIC_IMAGE_ALIGNMENT,
                                                  config->debug_key_frame_synetci_img_alignment_mutex);
        check_and_change_config_according_to_menu(menuDebugPixelSelection, config->DEBUG_PIXEL_SELECTION,
                                                  config->debug_pixel_selection_mutex);

        check_and_change_config_according_to_menu(menuDebugImagePyramid, config->DEBUG_IMAGE_PYRAMID,
                                                  config->debug_image_pyramid_mutex);

        check_and_change_config_according_to_menu(menuDebugCoarseToFineTracking, config->DEBUG_COARSE_TO_FINE_TRACKING,
                                                  config->debug_coarse_to_fine_tracking_mutex);
    }
}

inline void check_and_change_config_according_to_menu(pangolin::Var<bool>& menu_config, bool& project_config,
                                                      std::mutex& project_config_mutex) {
    if (menu_config && !project_config) {
        project_config_mutex.lock();
        if (menu_config) {
            project_config = true;
            project_config_mutex.unlock();
        }
    } else if (!menu_config && project_config) {
        project_config_mutex.lock();
        if (!menu_config) {
            project_config = false;
            project_config_mutex.unlock();
        }
    }
}

void Visualizer::publish_curr_frame_img(cv::Mat img, const Sophus::SE3f& T_w_c_) {
    std::lock_guard<std::mutex> lg_curr_frame(curr_frame_mutex);
    this->curr_frame_img = img;
    this->T_w_c = T_w_c_;
    this->curr_frame_image_changed = true;
}

void Visualizer::publish_key_frame_depth(cv::Mat img) {
    std::lock_guard<std::mutex> lg_key_frame_depth(key_frame_depth_mutex);
    this->key_frame_depth = img;
    this->key_frame_depth_changed = true;
}

void Visualizer::draw_and_publish_curr_frame(std::shared_ptr<PointCloudPyramid> pcp, cv::Mat img,
                                             const Sophus::SE3f& T_w_c_) {
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
    publish_curr_frame_img(resize(result), T_w_c_);
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

void Visualizer::draw_curr_frame_cam() {
    Sophus::SE3f curr_pose;
    curr_frame_mutex.lock();
    curr_pose = T_w_c;
    curr_frame_mutex.unlock();

    draw_cam(curr_pose);
}

void Visualizer::draw_cam(const Sophus::SE3f& T_w_c_) {
    glPushMatrix();
    glMultMatrixf(T_w_c_.matrix().data());

    glColor3f(0, 1.0f, 0);

    glLineWidth(2);
    glBegin(GL_LINES);

    glVertex3f(0, 0, 0);
    glVertex3f(tl(0), tl(1), tl(2));

    glVertex3f(0, 0, 0);
    glVertex3f(tr(0), tr(1), tr(2));

    glVertex3f(0, 0, 0);
    glVertex3f(dl(0), dl(1), dl(2));

    glVertex3f(0, 0, 0);
    glVertex3f(dr(0), dr(1), dr(2));

    glVertex3f(tl(0), tl(1), tl(2));
    glVertex3f(tr(0), tr(1), tr(2));

    glVertex3f(tr(0), tr(1), tr(2));
    glVertex3f(dr(0), dr(1), dr(2));

    glVertex3f(dr(0), dr(1), dr(2));
    glVertex3f(dl(0), dl(1), dl(2));

    glVertex3f(dl(0), dl(1), dl(2));
    glVertex3f(tl(0), tl(1), tl(2));

    glEnd();
    glPopMatrix();
}

void Visualizer::stop_or_start_according_to_pangolin_menu() {
    std::unique_lock<std::mutex> ul(stop_main_thread_mutex);
    while (stop_main_thread) {
        ul.unlock();
        std::this_thread::sleep_for(std::chrono::seconds(2));
        ul.lock();
    }
}
}  // namespace mpl
