#include "visualizer.h"

namespace mpl {

void Visualizer::run() {
    pangolin::CreateWindowAndBind("MPL: Visualizer", 1024, 768);

    glEnable(GL_DEPTH_TEST);

    // add panel
    pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_W));
    pangolin::Var<bool> menuShowCurrentFrame("menu.Show Current Frame", true, true);

    // add Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(pangolin::ProjectionMatrix(W, H, 500, 500, W / 2, H / 2, 0.1, 1000),
                                      pangolin::ModelViewLookAt(0, -0.7, -1.8, 0, 0, 0, 0.0, -1.0, 0.0));

    // add 3D view(main)
    pangolin::View& view_3d = pangolin::CreateDisplay()
                                  .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_W), 1.0, -W / H)
                                  .SetHandler(new pangolin::Handler3D(s_cam));

    // add img view and set texture
    pangolin::View& view_curr_frame = pangolin::Display("curr frame").SetAspect(W / H);
    pangolin::View& view_key_frame = pangolin::Display("key frame").SetAspect(W / H);
    pangolin::View& view_key_frame_depth = pangolin::Display("key frame depth").SetAspect(W / H);

    pangolin::GlTexture tex_view_curr_frame(W, H, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
    pangolin::GlTexture tex_view_key_frame(W, H, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
    pangolin::GlTexture tex_view_key_frame_depth(W, H, GL_RGBA, false, 0, GL_RGBA, GL_UNSIGNED_SHORT);

    pangolin::CreateDisplay()
        .SetBounds(0.0, 0.3, pangolin::Attach::Pix(UI_W), 1.0)
        .SetLayout(pangolin::LayoutEqual)
        .AddDisplay(view_curr_frame)
        .AddDisplay(view_key_frame)
        .AddDisplay(view_key_frame_depth);

    pangolin::OpenGlMatrix Twc;
    Twc.SetIdentity();

    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // display 3d

        // display image
        if (curr_frame_changed) tex_view_curr_frame.Upload(curr_frame_img.data, GL_RGB, GL_UNSIGNED_BYTE);
        if (key_frame_changed) tex_view_curr_frame.Upload(curr_frame_img.data, GL_RGB, GL_UNSIGNED_BYTE);
        if (key_frame_depth_changed) tex_view_curr_frame.Upload(curr_frame_img.data, GL_RGB, GL_UNSIGNED_BYTE);
        curr_frame_img = key_frame_depth_changed = key_frame_changed = false;

        view_curr_frame.Activate();
        glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
        tex_view_curr_frame.RenderToViewportFlipY();

        view_key_frame.Activate();
        glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
        tex_view_key_frame.RenderToViewportFlipY();

        view_key_frame_depth.Activate();
        glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
        tex_view_key_frame_depth.RenderToViewportFlipY();

        pangolin::FinishFrame();
    }
}

void Visualizer::publish_curr_frame_img(cv::Mat img) {
    this->curr_frame_img = img;
    this->curr_frame_changed = true;
}
void Visualizer::publish_key_frame_img(cv::Mat img) {
    this->key_frame_img = img;
    this->key_frame_changed = true;
}
void Visualizer::publish_key_frame_depth(cv::Mat img) {
    this->key_frame_depth = img;
    this->key_frame_depth_changed = true;
}
}  // namespace mpl
