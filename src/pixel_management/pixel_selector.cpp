#include "pixel_selector.h"
#include "../util/tictoc.h"
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <thread>
namespace mpl {
PixelSelector::PixelSelector() {
    config = &Config::getInstance();

    thresh_block_dim = config->THRESHOLD_BLOCK_DIM;

    cam = &CamData::getInstance();
    set_pot_dim(config->PIXEL_SELECTION_POT_DIM);

    thresh_block_num_x = ceil((float)cam->width[0] / thresh_block_dim);
    thresh_block_num_y = ceil((float)cam->height[0] / thresh_block_dim);

    thresh_map = std::unique_ptr<float[]>(new float[thresh_block_num_x * thresh_block_num_y]);
    thresh_map_backup = std::unique_ptr<float[]>(new float[thresh_block_num_x * thresh_block_num_y]);

    recursive_count = 0;

    candidates.reserve(2 * config->PIXEL_SELECTION_NUM);  // todo experiment check this num
}

int PixelSelector::select(Frame::ptr frame_, std::vector<Eigen::Vector3i>& candidates_out, cv::Mat depth_safe_mask_,
                          cv::Mat alignment_mask_) {
    this->depth_safe_mask = depth_safe_mask_;
    this->alignment_mask = alignment_mask_;

    if (this->candidates.size()) {
        reset();
    }

    tictoc::tic();

    this->frame = frame_;
    fill_thresh_map();
    filtering_thresh_map();

    bool is_finish = select_adaptively(this->pot_dim);

    int time_cost = tictoc::toc();
    if (is_finish) {
        std::move(candidates.begin(), candidates.end(), std::back_inserter(candidates_out));

        debug::execute_mem_according_to_config(config->DEBUG_PIXEL_SELECTION, config->debug_pixel_selection_mutex,
                                               &PixelSelector::draw_result, this, candidates_out,
                                               frame->get_synetic_photometric_pyramid(), time_cost);

        return this->candidates.size();
    }
    return 0;
}

void PixelSelector::draw_result(const std::vector<Eigen::Vector3i>& candidates, ImagePyramid::ptr pyramid_ptr,
                                const float time_cost) const {
    cv::Mat raw_im(cam->height[0], cam->width[0], CV_8UC1);
    raw_im.data = pyramid_ptr->data(0).get();
    cv::Mat result;
    cv::cvtColor(raw_im, result, CV_GRAY2BGR);

    for (const auto& can : candidates) {
        cv::Scalar color;
        if (can(2) == 0) {
            color = cv::Scalar(0, 255, 0);
        } else if (can(2) == 1) {
            color = cv::Scalar(255, 0, 0);
        } else {
            color = cv::Scalar(0, 0, 255);
        }

        cv::circle(result, cv::Point(can(0), can(1)), 1, color, 2);
    }
    cv::imshow("pixle selection", result);
    LOG(INFO) << "PIXEL SELECTION :" << this->candidates.size() << " is selected" << '\n'
              << "time cost : " << time_cost / 1000.0f << " ms";
    cv::waitKey(0);
}

void PixelSelector::fill_thresh_map() {
    for (int block_x = 0; block_x < thresh_block_num_x; ++block_x) {
        for (int block_y = 0; block_y < thresh_block_num_y; ++block_y) {
            // we sum each block the gradient magitude information

            int x_max = std::min((block_x + 1) * thresh_block_dim, cam->width[0]);
            int y_max = std::min((block_y + 1) * thresh_block_dim, cam->height[0]);
            std::vector<float> mag2_each_block;  // todo avoid allocate each time

            for (int x = block_x * thresh_block_dim; x < x_max; ++x) {
                for (int y = block_y * thresh_block_dim; y < y_max; ++y) {
                    float mag2_curr_pixel = frame->mag_squared_synetic(x, y);
                    mag2_each_block.push_back(std::move(mag2_curr_pixel));
                }
            }
            std::nth_element(mag2_each_block.begin(), mag2_each_block.begin() + mag2_each_block.size() / 2,
                             mag2_each_block.end());
            float median = mag2_each_block[mag2_each_block.size() / 2];
            thresh_map[idx(thresh_block_num_x, block_x, block_y)] =
                sqrt(median) + config->PIXEL_SELECTION_HERURISTIC_CONST;
        }
    }
}
// apply gauss filtering
void PixelSelector::filtering_thresh_map() {
    memmove(thresh_map_backup.get(), thresh_map.get(), thresh_block_num_x * thresh_block_num_y * sizeof(float));

    for (int block_x = 0; block_x < thresh_block_num_x; ++block_x) {
        for (int block_y = 0; block_y < thresh_block_num_y; ++block_y) {
            float sum = 0.0f;
            //	-	-	-			1	2	1
            //	-	x	-	with	2	4	2
            //	-	-	-			1	2	1
            for (int x = -1; x < 2; ++x) {
                for (int y = -1; y < 2; ++y) {
                    int safe_block_x = std::min(std::max(block_x + x, 0), thresh_block_num_x - 1);
                    int safe_block_y = std::min(std::max(block_y + y, 0), thresh_block_num_y - 1);
                    sum += weight[idx(3, x + 1, y + 1)] *
                           thresh_map_backup[idx(thresh_block_num_x, safe_block_x, safe_block_y)];
                }
            }
            thresh_map[idx(thresh_block_num_x, block_x, block_y)] = sum * sum;  // we compare with sum^2
        }
    }
}  // namespace mpl

void PixelSelector::reset() {
    if (pot_dim != config->PIXEL_SELECTION_POT_DIM) {
        set_pot_dim(config->PIXEL_SELECTION_POT_DIM);
    }
    candidates.clear();  // clear do not changet the vector capacity
    recursive_count = 0;
}

void PixelSelector::set_pot_dim(const int pot_dim_want) {
    pot_dim = pot_dim_want;
    block_dim = pot_dim * 2;
    grid_dim = block_dim * 2;
    grid_num_x = ceil((float)cam->width[0] / grid_dim);
    grid_num_y = ceil((float)cam->height[0] / grid_dim);
}

// recursive selection
bool PixelSelector::select_adaptively(int pot_dim_want) {
    // check return condition
    if (pot_dim_want != pot_dim) {
        // pot should at least has size 1
        pot_dim = std::max(pot_dim_want, 1);
        LOG(INFO) << " pot now : " << pot_dim_want;
        set_pot_dim(pot_dim_want);
        candidates.clear();
    } else if (candidates.size() != 0) {
        // if we have the same pot_size as before we return directly
        return true;
    }

    select_in_image();

    //
    int num_selected = candidates.size();
    int num_want = config->PIXEL_SELECTION_NUM;
    if ((num_selected > 0.875 * num_want && num_selected < 1.125 * num_want) ||
        recursive_count > config->PIXEL_SELECTION_RECURSIVE_LIMIT) {
        return true;
    } else {
        int pot_dim_old = pot_dim_want;
        int pot_dim_new = pot_dim_old / sqrt(num_want / (float)num_selected) + 1;
        ++recursive_count;
        return select_adaptively(pot_dim_new);
    }
}

void PixelSelector::select_with_grid_range(const int min_id, const int max_id) {
    for (int i = min_id; i < max_id; ++i) {
        const int grid_x = i % grid_num_x;
        const int grid_y = i / grid_num_x;
        select_in_one_grid(grid_x, grid_y);
    }
}

void PixelSelector::select_in_image() {
    bool multi_thread = true;

    if (multi_thread) {
        const int num_threads = 6;
        const int total_grid_num = grid_num_x * grid_num_y;
        const int num_grid_each_group = total_grid_num / num_threads;

        std::vector<std::thread> th_vec;
        for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
            const int grid_id_min = thread_id * num_grid_each_group;
            const int grid_id_max = grid_id_min + num_grid_each_group;
            th_vec.emplace_back(&PixelSelector::select_with_grid_range, std::ref(*this), grid_id_min, grid_id_max);
        }

        // remain grid
        if (total_grid_num % num_threads) {
            th_vec.emplace_back(&PixelSelector::select_with_grid_range, std::ref(*this),
                                num_threads * num_grid_each_group, total_grid_num);
        }

        for (auto& th : th_vec) {
            th.join();
        }

    } else {
        for (int grid_x = 0; grid_x < grid_num_x; ++grid_x) {
            for (int grid_y = 0; grid_y < grid_num_y; ++grid_y) {
                select_in_one_grid(grid_x, grid_y);
            }
        }
    }
}

inline bool PixelSelector::is_valid(const int u, const int v) const {
    if (!(this->depth_safe_mask.at<float>(v, u) < 1e-10 && ((int)this->alignment_mask.at<uchar>(v, u) < 10))) {
        return false;
    }
    return true;

    // const float dx = frame->dx(u, v);
    // const float dy = frame->dy(u, v);
    // const float mag2 = frame->mag_squared(u, v);
    //
    //    const float syn_dx = frame->get_synetic_photometric_pyramid()->dx(u, v);
    //    const float syn_dy = frame->get_synetic_photometric_pyramid()->dy(u, v);
    //    const float syn_mag2 = frame->mag_squared_synetic(u, v);
    //    const float angle = std::acos((dx * syn_dx + dy * syn_dy) / (std::sqrt(mag2) * std::sqrt(syn_mag2)));
    //    // if (angle > 0.174f) {
    //    //    std::cout << "angle : " << angle << '\n';
    //    //}
    //
    //    return (angle < 0.2f && (dx + dy) > 1e-4);  // rad
}

void PixelSelector::select_in_one_grid(const int grid_x, const int grid_y) {
    bool is_one_block_success = false;
    // for each block in grid
    for (int block_x = 0; block_x < 2; ++block_x) {
        for (int block_y = 0; block_y < 2; ++block_y) {
            bool is_one_pot_success = false;
            // for each pot in block
            for (int pot_x = 0; pot_x < 2; ++pot_x) {
                for (int pot_y = 0; pot_y < 2; ++pot_y) {
                    Eigen::Vector3i candidate;
                    float mag2_max = find_max_mag2(candidate, grid_x, grid_y, block_x, block_y, pot_x, pot_y);
                    if (mag2_max > thresh_map[thresh_idx(candidate)]) {
                        candidates_mutex.lock();
                        candidates.push_back(candidate);
                        candidates_mutex.unlock();
                        is_one_pot_success = true;
                        is_one_block_success = true;
                    }
                }
            }
            if (!is_one_pot_success) {
                Eigen::Vector3i candidate;
                float mag2_max = find_max_mag2(candidate, grid_x, grid_y, block_x, block_y);
                if (mag2_max > thresh_map[thresh_idx(candidate)] * config->PIXEL_SELECTION_DOWNWEIGHT) {
                    candidates_mutex.lock();
                    candidates.push_back(candidate);
                    candidates_mutex.unlock();
                    is_one_block_success = true;
                }
            }
        }
    }
    if (!is_one_block_success) {
        Eigen::Vector3i candidate;
        float mag2_max = find_max_mag2(candidate, grid_x, grid_y);
        if (mag2_max > thresh_map[thresh_idx(candidate)] * config->PIXEL_SELECTION_DOWNWEIGHT *
                           config->PIXEL_SELECTION_DOWNWEIGHT) {
            candidates_mutex.lock();
            candidates.push_back(candidate);
            candidates_mutex.unlock();
        }
    }
}

float PixelSelector::find_max_mag2(Eigen::Vector3i& candidate, const int grid_x, const int grid_y, const int block_x,
                                   const int block_y, const int pot_x, const int pot_y) {
    SearchRegion search_region;
    if (pot_x != -1 && pot_y != -1) {
        search_region = SearchRegion::POT_LVL;
    } else if (block_x != -1 && block_y != -1) {
        search_region = SearchRegion::BLOCK_LVL;
    } else {
        search_region = SearchRegion::GRID_LVL;
    }

    switch (search_region) {
        case SearchRegion::POT_LVL: {
            const int tl_x = grid_x * grid_dim + block_x * block_dim + pot_x * pot_dim;
            const int tl_y = grid_y * grid_dim + block_y * block_dim + pot_y * pot_dim;

            const int pot_right_bound = tl_x + pot_dim;
            const int pot_lower_bound = tl_y + pot_dim;
            if (!(pot_right_bound < cam->width[0]) || !(pot_lower_bound < cam->height[0])) {
                return 0;
            }

            int max_x = -1, max_y = -1;
            float max_mag2 = 0.0f;
            for (int y = 0; y < pot_dim; ++y) {
                for (int x = 0; x < pot_dim; ++x) {
                    const int global_x = tl_x + x;
                    const int global_y = tl_y + y;
                    const float curr_mag2 = frame->mag_squared_synetic(global_x, global_y);
                    if (curr_mag2 > max_mag2 && is_valid(global_x, global_y)) {
                        max_x = global_x;
                        max_y = global_y;
                        max_mag2 = curr_mag2;
                    }
                }
            }

            candidate(0) = max_x;
            candidate(1) = max_y;
            candidate(2) = 0;

            return max_mag2;
        } break;
        case SearchRegion::BLOCK_LVL: {
            const int tl_x = grid_x * grid_dim + block_x * block_dim;
            const int tl_y = grid_y * grid_dim + block_y * block_dim;

            const int block_right_bound = tl_x + block_dim;
            const int block_lower_bound = tl_y + block_dim;
            if (!(block_right_bound < cam->width[0]) || !(block_lower_bound < cam->height[0])) {
                return 0;
            }

            int max_x = -1, max_y = -1;
            float max_mag2 = 0;
            for (int y = 0; y < block_dim; ++y) {
                for (int x = 0; x < block_dim; ++x) {
                    const int global_x = tl_x + x;
                    const int global_y = tl_y + y;
                    const float curr_mag2 = frame->mag_squared_synetic(global_x, global_y);
                    if (curr_mag2 > max_mag2 && is_valid(global_x, global_y)) {
                        max_x = global_x;
                        max_y = global_y;
                        max_mag2 = curr_mag2;
                    }
                }
            }

            candidate(0) = max_x;
            candidate(1) = max_y;
            candidate(2) = 1;

            return max_mag2;

        } break;
        case SearchRegion::GRID_LVL: {
            // if there homogenous area in image bound, igonore it will be also
            // rational
            const int tl_x = grid_x * grid_dim;
            const int tl_y = grid_y * grid_dim;

            int grid_right_bound = tl_x + grid_dim;
            int grid_lower_bound = tl_y + grid_dim;

            if (!(grid_right_bound < cam->width[0]) || !(grid_lower_bound < cam->height[0])) {
                return 0;
            }

            int max_x = -1, max_y = -1;
            float max_mag2 = 0.0f;
            // copy to a vector to use max_element
            for (int y = 0; y < grid_dim; ++y) {
                for (int x = 0; x < grid_dim; ++x) {
                    const int global_x = tl_x + x;
                    const int global_y = tl_y + y;
                    const float curr_mag2 = frame->mag_squared_synetic(global_x, global_y);
                    if (curr_mag2 > max_mag2 && is_valid(global_x, global_y)) {
                        max_x = global_x;
                        max_y = global_y;
                        max_mag2 = curr_mag2;
                    }
                }
            }
            candidate(0) = max_x;
            candidate(1) = max_y;
            candidate(2) = 2;

            return max_mag2;
        } break;

        default:
            break;
    }
    return 0;
}
}  // namespace mpl
