#include "visualizer.h"
#include "cam_data.h"
#include "frame.h"
#include "point_cloud_pyramid.h"
#include <opencv2/imgproc.hpp>

namespace mpl {

void Visualizer::init() {
    // img data
    cam_data = &CamData::getInstance();
    img_cols = cam_data->width[scale_factor];
    img_rows = cam_data->height[scale_factor];

    W = cam_data->width[0] + UI_W;
    H = cam_data->height[0] * 2;

    pangolin::CreateWindowAndBind("MPL: Visualizer", W, H);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // add panel
    pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_W));

    // add Camera Render Object (for view / scene browsing)
    cam_3d = pangolin::OpenGlRenderState(pangolin::ProjectionMatrix(W, H, 500, 500, W / 2.0f, H / 2.0f, 0.1f, 1000.0f),
                                         pangolin::ModelViewLookAt(0, -0.7, -1.8, 0, 0, 0, 0.0, -1.0, 0.0));

    // add 3D view(main)
    view_3d = pangolin::CreateDisplay()
                  .SetBounds(0.0, 1.0f, pangolin::Attach::Pix(UI_W), 1.0f, -W / H)
                  .SetHandler(new pangolin::Handler3D(cam_3d));
}
void Visualizer::run() {
    init();

    // menu
    pangolin::Var<bool> menuShowCurrentFrame("menu.Show Current Frame", true, true);

    // add img view and set texture
    pangolin::View& view_curr_frame = pangolin::Display("curr frame")
                                          .SetAspect(img_cols / img_rows)
                                          .SetLock(pangolin::LockLeft, pangolin::LockBottom);

    pangolin::View& view_key_frame_depth = pangolin::Display("key frame depth")
                                               .SetAspect(img_cols / img_rows)
                                               .SetLock(pangolin::LockRight, pangolin::LockBottom);

    pangolin::GlTexture tex_view_curr_frame(img_cols, img_rows, GL_RGBA8, false, 0, GL_RGBA, GL_UNSIGNED_BYTE);
    pangolin::GlTexture tex_view_key_frame_depth(img_cols, img_rows, GL_RGBA, false, 0, GL_RGBA, GL_UNSIGNED_SHORT);

    pangolin::CreateDisplay()
        .SetBounds(0.0, 0.3, pangolin::Attach::Pix(UI_W), 1.0)
        .AddDisplay(view_curr_frame)
        .AddDisplay(view_key_frame_depth)
        .SetLayout(pangolin::LayoutEqual);

    pangolin::OpenGlMatrix Twc;
    Twc.SetIdentity();

    while (true) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // display 3d
        view_3d.Activate(cam_3d);
        glColor3f(1.0, 1.0, 1.0);
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
                tex_view_key_frame_depth.Upload(key_frame_depth.data, GL_BGRA, GL_UNSIGNED_SHORT);
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
    cv::Mat tmp;
    cv::cvtColor(img, tmp, cv::COLOR_GRAY2BGRA);
    tmp = resize(tmp);

    std::lock_guard<std::mutex> lg_key_frame_depth(key_frame_depth_mutex);
    this->key_frame_depth = tmp;
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

inline cv::Mat Visualizer::resize(cv::Mat im) const {
    cv::resize(im, im, cv::Size(im.cols >> scale_factor, im.rows >> scale_factor));
    return im;
}

}  // namespace mpl
