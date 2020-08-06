#include "visualizer.h"
#include "cam_data.h"
#include "candidate_manager.h"
#include "frame.h"
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
                                         pangolin::ModelViewLookAt(2, -2, -12, 0, 0, 0, pangolin::AxisNegY));

    // add 3D view(main)
    view_3d = pangolin::CreateDisplay()
                  .SetBounds(0.25, 1.0, pangolin::Attach::Pix(UI_W), 1.0, -W / H)
                  .SetHandler(new pangolin::Handler3D(cam_3d));

    // data to draw the camera frame
    float scale = 0.9f;
    tl = unproject(cam_data, Eigen::Vector2i(0, 0), 1 / scale);
    tr = unproject(cam_data, Eigen::Vector2i(img_cols - 1, 0), 1 / scale);
    dl = unproject(cam_data, Eigen::Vector2i(0, img_rows - 1), 1 / scale);
    dr = unproject(cam_data, Eigen::Vector2i(img_cols - 1, img_rows - 1), 1 / scale);
}
void Visualizer::run() {
    init();

    // menu
    pangolin::Var<bool> menuStop("menu.Stop", true, true);
    pangolin::Var<bool> menuDebugDistanceMap("menu.Debug Distance Map", false, true);
    pangolin::Var<bool> menuShowKeyFrameSyneticImageAlignment("menu.Debug Image Alignment", false, true);
    pangolin::Var<bool> menuDebugPixelSelection("menu.Debug Pixel Selection", false, true);
    pangolin::Var<bool> menuDebugImagePyramid("menu.Debug Image Pydamid", false, true);
    pangolin::Var<bool> menuDebugCoarseToFineTracking("menu.Debug Pyramid Tracking", false, true);
    pangolin::Var<bool> menuDebugDepthSafeMask("menu.Debug Depth Safe Mask", false, true);

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
        draw_sliding_window();
        draw_tracking_point_cloud();
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
        check_and_change_config_according_to_menu(menuDebugDepthSafeMask, config->DEBUG_DEPTH_SAFE_MASK,
                                                  config->debug_depth_safe_mask_mutex);
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

void Visualizer::set_curr_frame_tracking_info(cv::Mat img, const Sophus::SE3f& T_w_c_, const Sophus::SE3f& T_c_kf) {
    std::lock_guard<std::mutex> lg_curr_frame(curr_frame_mutex);
    this->curr_frame_img = img;
    this->T_w_c = T_w_c_;
    this->curr_frame_image_changed = true;
    this->T_c_kf = T_c_kf;
}

void Visualizer::set_key_frame_depth(cv::Mat img) {
    std::lock_guard<std::mutex> lg_key_frame_depth(key_frame_depth_mutex);
    this->key_frame_depth = img;
    this->key_frame_depth_changed = true;
}

void Visualizer::publish_curr_frame_tracking_info(std::shared_ptr<PointCloudPyramid> pcp, cv::Mat img,
                                                  const Sophus::SE3f& T_w_c_, const Sophus::SE3f& T_c_kf,
                                                  const bool draw_outlier) {
    // compute statistic
    curr_frame_mutex.lock();
    this->pcd.resize(pcp->operator[](0).size());
    std::copy(pcp->operator[](0).begin(), pcp->operator[](0).end(), pcd.begin());
    curr_frame_mutex.unlock();

    int point_size = pcd.size();
    std::vector<float> valid_depth_vec;
    valid_depth_vec.reserve(point_size);

    for (const auto& point : pcd) {
        if (!point.vis_data.visible_for_newst_frame) continue;
        if (isfinite(point.vis_data.depth_in_newst_frame) && point.vis_data.depth_in_newst_frame > 1e-10) {
            valid_depth_vec.push_back(point.vis_data.depth_in_newst_frame);
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

    for (const auto& point : pcd) {
        if (!point.vis_data.visible_for_newst_frame || point.vis_data.is_outlier) continue;
        Eigen::Vector2f hit_pixel = point.vis_data.hit_pixel_in_newst_frame;

        float depth = point.vis_data.depth_in_newst_frame;
        if (!isfinite(depth) || depth < 1e-10) {
            continue;
        } else {
            depth = std::max(std::min(depth, hi), lo);
        }

        uchar wrapped_depth = 255 * std::pow((depth - lo) / (hi - lo), 0.9f);

        cv::circle(depth_map, cv::Point2f(hit_pixel(0), hit_pixel(1)), 1, cv::Scalar(wrapped_depth), 2);
    }
    cv::Mat mask = depth_map.clone();
    cv::applyColorMap(depth_map, depth_map, cv::COLORMAP_JET);
    depth_map.copyTo(result, mask);

    // draw outlier
    if (draw_outlier) {
        for (const auto& point : pcd) {
            if (!point.vis_data.visible_for_newst_frame) continue;
            Eigen::Vector2f hit_pixel = point.vis_data.hit_pixel_in_newst_frame;
            if (point.vis_data.is_outlier) {
                cv::circle(result, cv::Point2f(hit_pixel(0), hit_pixel(1)), 1, cv::Scalar(255, 255, 255), 2);
            }
        }
    }

    cv::cvtColor(result, result, cv::COLOR_BGR2RGBA);
    set_curr_frame_tracking_info(resize(result), T_w_c_, T_c_kf);
}

void Visualizer::draw_tracking_point_cloud() {
    this->curr_frame_mutex.lock();
    std::vector<Voxel> point_cloud(pcd.size());
    std::copy(pcd.begin(), pcd.end(), point_cloud.begin());
    const Sophus::SE3f T_w_kf = this->T_w_c * this->T_c_kf;
    this->curr_frame_mutex.unlock();

    glPointSize(POINT_SIZE);
    glBegin(GL_POINTS);

    glColor4f(0.0f, 1.0f, 0.0f, 1.0f);
    for (auto& point : pcd) {
        if (!point.vis_data.visible_for_newst_frame) continue;
        Eigen::Vector3f P_w = T_w_kf * point.position;
        glVertex3f(P_w(0), P_w(1), P_w(2));
    }
    glEnd();
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
            } else {
                depth = std::min(std::max((float)depth, lo), hi);
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

    set_key_frame_depth(resize(result));
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

    glColor3f(1.0f, 0, 0);

    glLineWidth(3);
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
void Visualizer::draw_sliding_window() {
    std::vector<Sophus::SE3f> T_w_c_vec(this->T_w_c_sliding_window.size());
    std::vector<Eigen::Vector3f> point_vec(this->point_cloud_siding_window.size());

    this->sliding_window_mutex.lock();
    std::copy(this->T_w_c_sliding_window.begin(), this->T_w_c_sliding_window.end(), T_w_c_vec.begin());
    std::copy(this->point_cloud_siding_window.begin(), this->point_cloud_siding_window.end(), point_vec.begin());
    this->sliding_window_mutex.unlock();
    // draw all frame
    draw_cameras(T_w_c_vec);
    // draw all point
    draw_point_clouds(point_vec, POINT_SIZE, 0.0f, 0.0f, 1.0f);
}

void Visualizer::draw_cameras(const std::vector<Sophus::SE3f>& pose_vec) {
    // draw camera
    std::vector<line> lines;

    Eigen::Vector3f last_center(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < pose_vec.size(); ++i) {
        this->draw_cam(pose_vec[i]);

        if (i > 0) {
            lines.push_back({pose_vec[i].translation(), last_center});
        }
        last_center = pose_vec[i].translation();
    }

    // draw conneted line
    draw_lines(lines, 5.0f, 1.0f, 0.0f, 1.0f);
}

void Visualizer::draw_lines(const std::vector<line>& line_vec, const float line_width, const float r, const float g,
                            const float b, const Sophus::SE3f& T_w_o) {
    glPushMatrix();
    glMultMatrixf(T_w_o.matrix().data());

    glColor3f(r, g, b);

    glLineWidth(line_width);
    glBegin(GL_LINES);

    for (const auto& line : line_vec) {
        glVertex3f(line.first(0), line.first(1), line.first(2));
        glVertex3f(line.second(0), line.second(1), line.second(2));
    }

    glEnd();
    glPopMatrix();
}

void Visualizer::draw_point_clouds(const std::vector<Eigen::Vector3f>& point_vec, const float point_size, const float r,
                                   const float g, const float b) {
    glPointSize(point_size);
    glBegin(GL_POINTS);
    glColor4f(r, g, b, 1.0f);
    for (auto& point : point_vec) {
        glVertex3f(point(0), point(1), point(2));
    }
    glEnd();
}

void Visualizer::stop_or_start_according_to_pangolin_menu() {
    std::unique_lock<std::mutex> ul(stop_main_thread_mutex);
    while (stop_main_thread) {
        ul.unlock();
        std::this_thread::sleep_for(std::chrono::seconds(2));
        ul.lock();
    }
}

void Visualizer::set_new_sliding_window_data(CandidateManager& cm, const Frame::ptr to_remove_frame) {
    const auto& key_frame_vec = cm.get_key_frames();
    auto& candidate_map = cm.get_candidate_map();

    // prepare data
    std::vector<Sophus::SE3f> pose_buff_sliding_window;
    std::vector<Eigen::Vector3f> point_buff_sliding_window;
    std::vector<Eigen::Vector3f> point_buff_history;
    for (auto it = key_frame_vec.begin(); it != key_frame_vec.end(); it++) {
        const auto pose = ((*it)->get_pose());
        pose_buff_sliding_window.push_back(pose);
        auto& can_vec = candidate_map[*it];
        for (const auto& can : can_vec) {
            if (can.status == CandidateStatus::ACTIVE || can.status == CandidateStatus::OOB) {
                Eigen::Vector3f point_3d = pose * (*it)->unproject(Eigen::Vector2i(can.u, can.v), can.d_inv_synetic_im);

                point_buff_sliding_window.push_back(point_3d);

                if (*it == to_remove_frame) {
                    point_buff_history.push_back(point_3d);
                }
            }
        }
    }
    // set internal data
    sliding_window_mutex.lock();

    this->T_w_c_sliding_window.resize(pose_buff_sliding_window.size());
    this->point_cloud_siding_window.resize(point_buff_sliding_window.size());

    std::move(pose_buff_sliding_window.begin(), pose_buff_sliding_window.end(), this->T_w_c_sliding_window.begin());
    std::move(point_buff_sliding_window.begin(), point_buff_sliding_window.end(),
              this->point_cloud_siding_window.begin());
    sliding_window_mutex.unlock();

    if (to_remove_frame != nullptr) {
        history_mutex.lock();
        T_w_c_history.push(to_remove_frame->get_pose());
        this->point_cloud_history.push(point_buff_history);
        history_mutex.unlock();
    }
}
}  // namespace mpl
