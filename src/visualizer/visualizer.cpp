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
                  .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_W), 1.0, -W / H)
                  .SetHandler(new pangolin::Handler3D(cam_3d));
}
void Visualizer::run() {
    init();

    // menu
    pangolin::Var<bool> menuShowCurrentFrame("menu.Show Current Frame", true, true);

    // add img view and set texture
    pangolin::View& view_curr_frame = pangolin::Display("curr frame").SetAspect(img_cols / img_rows);
    pangolin::View& view_key_frame_depth = pangolin::Display("key frame depth").SetAspect(img_cols / img_rows);

    pangolin::GlTexture tex_view_curr_frame(img_cols, img_rows, GL_RGBA8, false, 0, GL_RGBA, GL_UNSIGNED_BYTE);
    pangolin::GlTexture tex_view_key_frame_depth(img_cols, img_rows, GL_RGBA, false, 0, GL_RGBA, GL_UNSIGNED_BYTE);

    pangolin::CreateDisplay()
        .SetBounds(0.0, 0.3, pangolin::Attach::Pix(UI_W), 1.0)
        .SetLayout(pangolin::LayoutEqual)
        .AddDisplay(view_curr_frame)
        .AddDisplay(view_key_frame_depth);

    pangolin::OpenGlMatrix Twc;
    Twc.SetIdentity();

    while (true) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // disp3ay 3d
        view_3d.Activate(cam_3d);
        glColor4f(1.0, 1.0, 1.0, 1.0f);
        pangolin::glDrawColouredCube();

        // display image
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
    assert(img.channels() == 4);
    this->key_frame_depth_changed = true;
}
void Visualizer::draw_and_publish_curr_frame(std::shared_ptr<PointCloudPyramid> pcp, cv::Mat img,
                                             const Sophus::SE3f& T_curr_KF) {
    cv::Mat to_draw = img.clone();

    for (auto& pcd : pcp->operator[](0)) {
        Eigen::Vector3f point = T_curr_KF * pcd.position;
        Eigen::Vector2f hit_pixel = project(cam_data, point);
        cv::circle(to_draw, cv::Point2f(hit_pixel(0), hit_pixel(1)), 1, cv::Scalar(0, 255, 0), 2);
    }

    cv::cvtColor(to_draw, to_draw, cv::COLOR_BGR2BGRA);
    publish_curr_frame_img(resize(to_draw));
}

void Visualizer::draw_and_publish_key_frame_depth(cv::Mat depth_im) {
    cv::Mat depth_im_8UC1 = cv::Mat::zeros(depth_im.size(), CV_8UC1);
    cv::Mat mask = cv::Mat(depth_im.size(), CV_8U, cv::Scalar(255));  // 0 indicate invalid pixel

    std::vector<ushort> valid_depth_vec;
    // compute lo and hi range of valid pixel
    for (int r = 0; r < depth_im.rows; ++r) {
        for (int c = 0; c < depth_im.cols; ++c) {
            ushort depth = depth_im.at<ushort>(r, c);
            if (std::isfinite(depth) && depth != 0) valid_depth_vec.push_back(depth);
        }
    }

    const float lo_percent = 0.01f;
    const float hi_percent = 0.99f;

    std::nth_element(valid_depth_vec.begin(), valid_depth_vec.begin() + lo_percent * valid_depth_vec.size(),
                     valid_depth_vec.end());
    const ushort lo = valid_depth_vec[lo_percent * valid_depth_vec.size()];

    std::nth_element(valid_depth_vec.begin(), valid_depth_vec.begin() + hi_percent * valid_depth_vec.size(),
                     valid_depth_vec.end());

    const ushort hi = valid_depth_vec[hi_percent * valid_depth_vec.size()];

    for (int r = 0; r < depth_im.rows; ++r) {
        for (int c = 0; c < depth_im.cols; ++c) {
            ushort depth = depth_im.at<ushort>(r, c);
            if (!std::isfinite(depth) || depth == 0) {
                mask.at<uchar>(r, c) = 0;
                continue;
            } else if (depth < lo) {
                depth_im_8UC1.at<uchar>(r, c) = 255;
            } else if (depth > hi) {
                depth_im_8UC1.at<uchar>(r, c) = 0;
            } else {
                depth_im_8UC1.at<uchar>(r, c) = 255 - 255 * (depth - lo) / float(hi - lo);
            }
        }
    }

    cv::Mat color_img;
    cv::applyColorMap(depth_im_8UC1, color_img, cv::COLORMAP_JET);

    cv::Mat result;
    color_img.copyTo(result, mask);

    cv::cvtColor(result, result, cv::COLOR_BGR2BGRA);

    publish_key_frame_depth(resize(result));
}

inline cv::Mat Visualizer::resize(cv::Mat im) const {
    cv::resize(im, im, cv::Size(im.cols >> scale_factor, im.rows >> scale_factor));
    return im;
}

}  // namespace mpl
