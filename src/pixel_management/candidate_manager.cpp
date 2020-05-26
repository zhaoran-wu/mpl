#include "candidate_manager.h"

namespace mpl {

void CandidateManager::update_depth_per_frame(const Frame::ptr frame,
                                              bool is_KF) {
    if (this->candidate_map.empty()) {
        return;
    }

    for (auto it = candidate_map.begin(); it != candidate_map.end(); ++it) {
        std::vector<Candidate> candi_per_frame = it->second;
        for (const Candidate& candi : candi_per_frame) {
            Eigen::Vector2f hit_pixel = unproject_trans_project(
                candi.d_inv, candi.uv, it->first, frame);

            if (!frame->is_in_image(0, candi.uv(0), candi.uv(1))) {
                continue;
            }
            std::pair<Candidate, Eigen::Vector2f> corres =
                find_corres(frame, candi);
            if (!is_outlier) {
                update_depth_per_pixel(corres, it->first, frame);
            }
        }
    }
}
}  // namespace mpl