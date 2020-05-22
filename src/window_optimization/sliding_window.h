#pragma once
#include "affine_light.h"
#include "candidate_manager.h"
#include "config.h"
#include "frame.h"
#include "sophus/se3.hpp"
#include "window_optimizer.h"
#include <Eigen/Core>
#include <array>
#include <glog/logging.h>
namespace mpl {

class SlidingWindow {
   public:
    SlidingWindow();
    void add_tracked_frame(const Frame::ptr frame,
                           std::vector<cv::Mat> synetic_im_vec);
    Frame::ptr get_lastKF() const;
    void optimize_window();

    void fix_origin(const Frame::ptr frame);
    /**
     * @brief get a semi-dense depth map for tracking
     *
     */
    std::vector<Candidate> get_depthmap();
    bool empty();

   private:
    std::array<Frame::ptr, Config::WINDOW_SIZE> KF_window;
    WindowOptimizer optimizer;
    CandidateManager candidate_manager;
};

inline Frame::ptr SlidingWindow::get_lastKF() const {
    auto iter =
        std::find_if(KF_window.rbegin(), KF_window.rend(),
                     [](Frame::ptr frame_ptr) { return frame_ptr != nullptr; });

    assert(iter != KF_window.rend() && *iter != nullptr);
    return *iter;
}
}  // namespace mpl