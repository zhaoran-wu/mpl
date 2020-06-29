#include "pixel_selector.h"
#include "../util/tictoc.h"
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
namespace mpl {
PixelSelector::PixelSelector() {
    config = &Config::getInstance();

    thresh_block_dim = config->THRESHOLD_BLOCK_DIM;

    cam = &CamData::getInstance();
    set_pot_dim(config->PIXEL_SELECTION_POT_DIM);

    thresh_block_num_x = ceil((float)cam->width[0] / thresh_block_dim);
    thresh_block_num_y = ceil((float)cam->height[0] / thresh_block_dim);

    thresh_map = std::unique_ptr<float[]>(
        new float[thresh_block_num_x * thresh_block_num_y]);
    thresh_map_backup = std::unique_ptr<float[]>(
        new float[thresh_block_num_x * thresh_block_num_y]);

    recursive_count = 0;

    candidates.reserve(
        2 * config->PIXEL_SELECTION_NUM);  // todo experiment check this num
}

int PixelSelector::select(ImagePyramid::ptr pyramid_ptr,
                          std::vector<Eigen::Vector3i>& candidates_out) {
    LOG_ASSERT(pyramid_ptr != nullptr);
    if (this->candidates.size()) {
        reset();
    }
    tictoc::tic();
    this->mag2_ptr = pyramid_ptr->mag2(0);

    fill_thresh_map();
    filtering_thresh_map();

    bool is_finish = select_adaptively(this->pot_dim);
    int time_cost = tictoc::toc();
    if (is_finish) {
        std::move(candidates.begin(), candidates.end(),
                  std::back_inserter(candidates_out));

        LOG(INFO) << "PIXLE SELECTION :" << this->candidates.size()
                  << " is selected" << '\n'
                  << "time cost : " << time_cost / 1000.0f << " ms";
        return this->candidates.size();
    }
    return 0;
}

void PixelSelector::fill_thresh_map() {
    float* mag2_map = mag2_ptr.get();

    for (int block_x = 0; block_x < thresh_block_num_x; ++block_x) {
        for (int block_y = 0; block_y < thresh_block_num_y; ++block_y) {
            // we sum each block the gradient magitude information

            int x_max =
                std::min((block_x + 1) * thresh_block_dim, cam->width[0]);
            int y_max =
                std::min((block_y + 1) * thresh_block_dim, cam->height[0]);
            std::vector<float>
                mag2_each_block;  // todo avoid allocate each time

            for (int x = block_x * thresh_block_dim; x < x_max; ++x) {
                for (int y = block_y * thresh_block_dim; y < y_max; ++y) {
                    float mag2_curr_pixel = mag2_map[idx(cam->width[0], x, y)];
                    mag2_each_block.push_back(std::move(mag2_curr_pixel));
                }
            }
            // todo extract as mpl::median
            std::nth_element(
                mag2_each_block.begin(),
                mag2_each_block.begin() + mag2_each_block.size() / 2,
                mag2_each_block.end());
            float median = mag2_each_block[mag2_each_block.size() / 2];
            thresh_map[idx(thresh_block_num_x, block_x, block_y)] =
                sqrt(median) + config->PIXEL_SELECTION_HERURISTIC_CONST;
        }
    }
}
// apply gauss filtering
void PixelSelector::filtering_thresh_map() {
    memmove(thresh_map_backup.get(), thresh_map.get(),
            thresh_block_num_x * thresh_block_num_y * sizeof(float));

    for (int block_x = 0; block_x < thresh_block_num_x; ++block_x) {
        for (int block_y = 0; block_y < thresh_block_num_y; ++block_y) {
            float sum = 0.0f;
            //	-	-	-			1	2	1
            //	-	x	-	with	2	4	2
            //	-	-	-			1	2	1
            for (int x = -1; x < 2; ++x) {
                for (int y = -1; y < 2; ++y) {
                    int safe_block_x = std::min(std::max(block_x + x, 0),
                                                thresh_block_num_x - 1);
                    int safe_block_y = std::min(std::max(block_y + y, 0),
                                                thresh_block_num_y - 1);
                    sum += weight[idx(3, x + 1, y + 1)] *
                           thresh_map_backup[idx(thresh_block_num_x,
                                                 safe_block_x, safe_block_y)];
                }
            }
            thresh_map[idx(thresh_block_num_x, block_x, block_y)] =
                sum * sum;  // we compare with sum^2
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
    resize_mag2_vec();
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
        int pot_dim_new =
            pot_dim_old / sqrt(num_want / (float)num_selected) + 1;
        ++recursive_count;
        return select_adaptively(pot_dim_new);
    }
}

void PixelSelector::select_in_image() {
    for (int grid_x = 0; grid_x < grid_num_x; ++grid_x) {
        for (int grid_y = 0; grid_y < grid_num_y; ++grid_y) {
            select_in_one_grid(grid_x, grid_y);
        }
    }
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
                    float mag2_max =
                        find_max_mag2(candidate, grid_x, grid_y, block_x,
                                      block_y, pot_x, pot_y);
                    if (mag2_max > thresh_map[thresh_idx(candidate)]) {
                        candidates.push_back(candidate);
                        is_one_pot_success = true;
                    }
                }
            }
            if (!is_one_pot_success) {
                Eigen::Vector3i candidate;
                float mag2_max =
                    find_max_mag2(candidate, grid_x, grid_y, block_x, block_y);
                if (mag2_max > thresh_map[thresh_idx(candidate)] *
                                   config->PIXEL_SELECTION_DOWNWEIGHT) {
                    candidates.push_back(candidate);
                    is_one_block_success = true;
                }
            }
        }
    }
    if (!is_one_block_success) {
        Eigen::Vector3i candidate;
        float mag2_max = find_max_mag2(candidate, grid_x, grid_y);
        if (mag2_max > thresh_map[thresh_idx(candidate)] *
                           config->PIXEL_SELECTION_DOWNWEIGHT *
                           config->PIXEL_SELECTION_DOWNWEIGHT) {
            candidates.push_back(candidate);
            is_one_block_success = true;
        }
    }
}

float PixelSelector::find_max_mag2(Eigen::Vector3i& candidate, const int grid_x,
                                   const int grid_y, const int block_x,
                                   const int block_y, const int pot_x,
                                   const int pot_y) {
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
            int pot_right_bound =
                grid_x * grid_dim + block_x * block_dim + (pot_x + 1) * pot_dim;
            int pot_lower_bound =
                grid_y * grid_dim + block_y * block_dim + (pot_y + 1) * pot_dim;
            if (!(pot_right_bound < cam->width[0]) ||
                !(pot_lower_bound < cam->height[0])) {
                return 0;
            }
            float* pot_head_ptr =
                get_pot_head(grid_x, grid_y, block_x, block_y, pot_x, pot_y);
            // copy to a vector to use max_element
            for (int y = 0; y < pot_dim; ++y) {
                memcpy(&pot_mag2_vec[y * pot_dim],
                       &pot_head_ptr[y * cam->width[0]],
                       pot_dim * sizeof(float));
            }
            auto max_iter =
                std::max_element(pot_mag2_vec.begin(), pot_mag2_vec.end());
            int distance = std::distance(pot_mag2_vec.begin(), max_iter);

            if (distance < 0) return 0;
            candidate(0) = grid_x * grid_dim + block_x * block_dim +
                           pot_x * pot_dim + distance % pot_dim;
            candidate(1) = grid_y * grid_dim + block_y * block_dim +
                           pot_y * pot_dim + distance / pot_dim;
            candidate(2) = 0;

            return *max_iter;
        } break;
        case SearchRegion::BLOCK_LVL: {
            int block_right_bound =
                grid_x * grid_dim + (block_x + 1) * block_dim;
            int block_lower_bound =
                grid_y * grid_dim + (block_y + 1) * block_dim;
            if (!(block_right_bound < cam->width[0]) ||
                !(block_lower_bound < cam->height[0])) {
                return 0;
            }
            float* block_head_ptr =
                get_block_head(grid_x, grid_y, block_x, block_y);
            // copy to a vector to use max_element
            for (int y = 0; y < block_dim; ++y) {
                memcpy(&block_mag2_vec[y * block_dim],
                       &block_head_ptr[y * cam->width[0]],
                       block_dim * sizeof(float));
            }
            auto max_iter =
                std::max_element(block_mag2_vec.begin(), block_mag2_vec.end());
            int distance = std::distance(max_iter, block_mag2_vec.begin());
            if (distance < 0) return 0;
            candidate(0) =
                grid_x * grid_dim + block_x * block_dim + distance % block_dim;
            candidate(1) =
                grid_y * grid_dim + block_y * block_dim + distance / block_dim;
            candidate(2) = 1;

            return *max_iter;
        } break;
        case SearchRegion::GRID_LVL: {
            // if there homogenous area in image bound, igonore it will be also
            // rational
            int grid_right_bound = (grid_x + 1) * grid_dim;
            int grid_lower_bound = (grid_y + 1) * grid_dim;
            if (!(grid_right_bound < cam->width[0]) ||
                !(grid_lower_bound < cam->height[0])) {
                return 0;
            }

            float* grid_head_ptr = get_grid_head(grid_x, grid_y);
            // copy to a vector to use max_element
            for (int y = 0; y < grid_dim; ++y) {
                memcpy(&grid_mag2_vec[y * grid_dim],
                       &grid_head_ptr[y * cam->width[0]],
                       grid_dim * sizeof(float));
            }
            auto max_iter =
                std::max_element(grid_mag2_vec.begin(), grid_mag2_vec.end());
            int distance = std::distance(max_iter, grid_mag2_vec.begin());

            if (distance < 0) return 0;
            if (candidate(0) < 0 || candidate(1) < 0) {
                candidate(0) = grid_x * grid_dim + distance % grid_dim;
                candidate(1) = grid_y * grid_dim + distance / grid_dim;
                candidate(2) = 2;
            }

            return *max_iter;
        } break;

        default:
            break;
    }
    return 0;
}

void PixelSelector::resize_mag2_vec() {
    pot_mag2_vec.resize(pot_dim * pot_dim, 0);
    block_mag2_vec.resize(4 * pot_dim * pot_dim, 0);
    grid_mag2_vec.resize(16 * pot_dim * pot_dim, 0);
}

}  // namespace mpl