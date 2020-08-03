#pragma once
#include "config.h"
#include "debug.h"
#include "frame.h"
#include "image_pyramid.h"
#include <Eigen/Core>
#include <memory>
#include <vector>

namespace mpl {

class PixelSelector {
    enum class SearchRegion {
        GRID_LVL,
        BLOCK_LVL,
        POT_LVL,
    };

   public:
    PixelSelector();

    /**
     * @brief interface to select candidate
     *
     * @param pyramid_ptr
     * @param candidates_out candidate_out(0):x , candidate_out(1):y ,
     * candidate_out(2): only for visulization, indicate in which searchRegion
     * ,it was selected
     * @param mask : candidate will not select if has a value of 0 in mask
     * @return int
     */
    int select(Frame::ptr frame, std::vector<Eigen::Vector3i>& candidates_out, cv::Mat depth_safe_mask,
               cv::Mat alignment_mask);

   private:
    // reset when new image to be selected
    void reset();
    void set_pot_dim(const int pot_dim_want);
    void fill_thresh_map();
    void filtering_thresh_map();
    // a control function to select adaptively with the enviroment,so that:
    // in texuture rich area we have a evenly distributed candidate
    // in homogenous area we make use of all possible pixel
    // return true if we get approximate num of candidates
    bool select_adaptively(int pot_dim_want);
    // select in whole images
    void select_in_image();
    // select in one grid
    void select_in_one_grid(const int grid_x, const int grid_y);

    float find_max_mag2(Eigen::Vector3i& candidate, const int grid_x, const int grid_y, const int block_x = -1,
                        const int block_y = -1, const int pot_x = -1, const int pot_y = -1);

    bool is_valid(const int u, const int v) const;
    int thresh_idx(const Eigen::Vector3i& pixel);
    void select_with_grid_range(const int min_id, const int max_id);  // thread function
    int thresh_block_dim;
    int thresh_block_num_x;
    int thresh_block_num_y;
    std::unique_ptr<float[]> thresh_map;  // each block has a thresh_vec
    std::unique_ptr<float[]> thresh_map_backup;
    const float weight[9] = {
        1.0f / 16, 2.0f / 16, 1.0f / 16,  //
        2.0f / 16, 4.0f / 16, 2.0f / 16,  //
        1.0f / 16, 2.0f / 16, 1.0f / 16,
    };  //

    // https://www.3dgep.com/wp-content/uploads/2011/11/Cuda-Execution-Model.png
    // here we mimic the nvidia name convention,
    // dim : micro lvl, width/height in terms of pixel, dim.x = dim.y = dim
    // num : macro lvl, each gird has 2x2 blocks,each block has 2*2 of pot
    int grid_dim;
    int block_dim;
    int pot_dim;

    int grid_num_x;  // each image contain n grid in x direction
    int grid_num_y;

    CamData* cam;
    Config* config;

    Frame::ptr frame;

    std::mutex candidates_mutex;
    std::vector<Eigen::Vector3i> candidates;

    int recursive_count;

    void draw_result(const std::vector<Eigen::Vector3i>& candidates, ImagePyramid::ptr pyramid_ptr,
                     const float time_cost) const;

    cv::Mat depth_safe_mask;
    cv::Mat alignment_mask;
};

//########################################################//
//#####################implementation#####################//

inline int PixelSelector::thresh_idx(const Eigen::Vector3i& pixel) {
    int x = (float)pixel(0) / thresh_block_dim;
    int y = (float)pixel(1) / thresh_block_dim;
    return idx(thresh_block_num_x, x, y);
}

}  // namespace mpl