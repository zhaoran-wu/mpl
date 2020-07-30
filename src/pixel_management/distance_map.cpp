#include "distance_map.h"
#include "candidate_manager.h"
#include <opencv2/imgproc.hpp>
namespace mpl {
DistanceMap::DistanceMap() {
    cam = &CamData::getInstance();
    w = cam->width[0];
    h = cam->height[0];

    // init
    this->require_update = (bool*)Eigen::internal::aligned_malloc(this->w * this->h * sizeof(bool));
    this->obstacles = (Eigen::Vector2i*)Eigen::internal::aligned_malloc(this->w * this->h * sizeof(Eigen::Vector2i));
    this->dist_map = (float*)Eigen::internal::aligned_malloc(this->w * this->h * sizeof(float));
}

DistanceMap::~DistanceMap() {
    Eigen::internal::aligned_free(this->require_update);
    Eigen::internal::aligned_free(this->obstacles);
    Eigen::internal::aligned_free(this->dist_map);
}

void DistanceMap::show_distance_map_for_visualization(bool normalize) {
    cv::Mat distTransform(this->h, this->w, CV_32FC1);
    std::copy(this->dist_map, this->dist_map + this->w * this->h, (float*)distTransform.data);

    if (normalize) {
        cv::normalize(distTransform, distTransform, 0, 1,
                      cv::NORM_MINMAX);               // normalize gray scale
        cv::pow(distTransform, 0.22, distTransform);  // pow scale
        cv::normalize(distTransform, distTransform, 0, 255, cv::NORM_MINMAX);

        distTransform.convertTo(distTransform, CV_8UC1);
        cv::applyColorMap(distTransform, distTransform, cv::COLORMAP_JET);
    }
    std::cout << "current active points num:   " << get_num_obstacles() << '\n';
    cv::namedWindow("dist_map", cv::WINDOW_AUTOSIZE);
    cv::imshow("dist_map", distTransform);
    cv::waitKey(0);
}

float DistanceMap::dist(int x, int y) const {
    return this->dist_map[x + (y * this->w)];
}

int DistanceMap::get_num_obstacles() const {
    return this->num_obstacles;
}

void DistanceMap::compute(std::unordered_map<Frame::ptr, std::vector<Candidate>>& candidate_map,
                          const Frame::ptr frame) {
    // reset distance map
    std::fill(this->dist_map, this->dist_map + this->w * this->h, std::numeric_limits<float>::max());
    this->num_obstacles = 0;

    // generate point in newst key frame and put dist = 0 in distance map
    for (auto& it : candidate_map) {
        if (it.first == frame) continue;

        //! go through all candidate
        auto& can_vec = it.second;
        for (auto& can : can_vec) {
            const Eigen::Vector2f point_in_frame = unproject_trans_project(can, it.first, frame);

            if (!is_in_img(*cam, point_in_frame)) {
                // can.status = (can.status == CandidateStatus::BAD) ? CandidateStatus::BAD : CandidateStatus::OOB;
                continue;
            }
            can.projection_on_newst_KF = point_in_frame;

            // check active point
            if (!can.is_active || can.age > 7 || can.status == CandidateStatus::BAD) continue;

            int x = static_cast<int>(point_in_frame[0]);
            int y = static_cast<int>(point_in_frame[1]);

            this->obstacles[++this->num_obstacles] = Eigen::Vector2i(x, y);
            this->require_update[this->num_obstacles] = true;
            this->dist_map[x + (y * this->w)] = 0;
        }
    }

    // go throught all obstacles propagating the distance outwards
    int radius = 1;
    bool updated = true;

    while (updated) {
        updated = false;

        for (int i = 0; i < this->num_obstacles; ++i) {
            if (!this->require_update[i]) continue;

            int x = this->obstacles[i].x();
            int y = this->obstacles[i].y();

            if (this->update_pixel_on_square_edge(x, y, radius)) {
                updated = true;
            } else {
                this->require_update[i] = false;
            }
        }

        radius++;
    }
}

void DistanceMap::add(const Eigen::Vector2f point_in_newst_KF) {
    // set obstacle
    int x = static_cast<int>(point_in_newst_KF[0]);
    int y = static_cast<int>(point_in_newst_KF[1]);

    int idx = x + (y * this->w);

    this->obstacles[this->num_obstacles] = Eigen::Vector2i(x, y);
    this->require_update[this->num_obstacles] = true;
    this->dist_map[idx] = 0;
    ++this->num_obstacles;

    // iterate in the new obstacle only
    int radius = 1;
    bool updated = true;

    while (updated) {
        updated = this->update_pixel_on_square_edge(x, y, radius);
        this->require_update[this->num_obstacles - 1] = updated;

        radius++;
    }
}

bool DistanceMap::update_pixel_on_square_edge(int x, int y, int radius) {
    bool updated = false;

    int u, v;

    // row -radius
    v = y - radius;
    if (v >= 0 && v < this->h)  // check that it is a valid row
    {
        // compute ranges
        int min_u = std::max(x - radius, 0);
        int max_u = std::min(x + radius, this->w - 1);

        for (u = min_u; u <= max_u; ++u) {
            float distX = static_cast<float>(u - x);
            float distY = static_cast<float>(v - y);

            // float dist = std::sqrtf(distX * distX + distY * distY);
            float dist = distX * distX + distY * distY;

            int idx = u + (v * this->w);

            if (this->dist_map[idx] > dist) {
                this->dist_map[idx] = dist;
                updated = true;
            }
        }
    }

    // row +radius
    v = y + radius;
    if (v >= 0 && v < this->h)  // check that it is a valid row
    {
        // compute ranges
        int min_u = std::max(x - radius, 0);
        int max_u = std::min(x + radius, this->w - 1);

        for (u = min_u; u <= max_u; ++u) {
            float distX = static_cast<float>(u - x);
            float distY = static_cast<float>(v - y);

            // float dist = std::sqrtf(distX * distX + distY * distY);
            float dist = distX * distX + distY * distY;

            int idx = u + (v * this->w);

            if (this->dist_map[idx] > dist) {
                this->dist_map[idx] = dist;
                updated = true;
            }
        }
    }

    // col -radius
    u = x - radius;
    if (u >= 0 && u < this->w)  // check that it is a valid col
    {
        // compute ranges
        int min_v = std::max(y - radius + 1, 0);
        int max_v = std::min(y + radius - 1, this->h - 1);

        for (v = min_v; v <= max_v; ++v) {
            float distX = static_cast<float>(u - x);
            float distY = static_cast<float>(v - y);

            // float dist = std::sqrtf(distX * distX + distY * distY);
            float dist = distX * distX + distY * distY;

            int idx = u + (v * this->w);

            if (this->dist_map[idx] > dist) {
                this->dist_map[idx] = dist;
                updated = true;
            }
        }
    }

    // col +radius
    u = x + radius;
    if (u >= 0 && u < this->w)  // check that it is a valid col
    {
        // compute ranges
        int min_v = std::max(y - radius + 1, 0);
        int max_v = std::min(y + radius - 1, this->h - 1);

        for (v = min_v; v <= max_v; ++v) {
            float distX = static_cast<float>(u - x);
            float distY = static_cast<float>(v - y);

            // float dist = std::sqrtf(distX * distX + distY * distY);
            float dist = distX * distX + distY * distY;

            int idx = u + (v * this->w);

            if (this->dist_map[idx] > dist) {
                this->dist_map[idx] = dist;
                updated = true;
            }
        }
    }

    return updated;
}
}  // namespace mpl