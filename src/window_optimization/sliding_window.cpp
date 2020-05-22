#include "sliding_window.h"
namespace mpl {

bool SlidingWindow::empty() {
    return (KF_window[0] == nullptr);
}

void SlidingWindow::fix_origin(const Frame::ptr frame) {
    assert(KF_window[0] == nullptr);
    KF_window[0] = frame;
    // todo send info to optimizer to fix origin
}
void SlidingWindow::add_tracked_frame(const Frame::ptr frame,
                                      std::vector<cv::Mat> synetic_im_vec) {
    assert(KF_window[0] != nullptr);
    bool is_KF = is_keyframe(frame);
    candidate_manager.update_depth_per_frame(frame, is_keyframe);

    if (!is_keyframe) {
        return;
    }
    candidate_manager.select_candidate(frame, synetic_im_vec[1]);
    LOG(INFO) << "key frame added";
    auto iter = std::find_if(KF_window.begin(), KF_window.end(), nullptr);
    bool is_KF_window_full = (iter == KF_window.end());
    if (is_KF_window_full) {
        Frame::ptr frame_to_marg = sw.get_frame_to_marg();
        auto iter_to_marg =
            std::find(KF_window.begin(), KF_window.end(), frame_to_marg);
        optimizer.marginallize(frame_to_marg);
        std::rotate(iter_to_marg, iter_to_marg + 1, KF_window.end());
        KF_window.back() = frame;

    } else {
        *iter = frame;
    }
}

}  // namespace mpl
