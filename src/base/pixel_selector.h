#pragma once
#include "config.h"
#include "image_pyramid.h"
#include <Eigen/Core>
#include <memory>
#include <vector>

namespace mpl {

class PixelSelector {
    enum SearchRegion {
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
     * @return int
     */
    int select(ImagePyramid::ptr pyramid_ptr,
               std::vector<Eigen::Vector3i>& candidates_out);

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

    float find_max_mag2(Eigen::Vector3i& candidate, const int grid_x,
                        const int grid_y, const int block_x = -1,
                        const int block_y = -1, const int pot_x = -1,
                        const int pot_y = -1);

    float* get_pot_head(const int grid_x, const int grid_y, const int block_x,
                        const int block_y, const int pot_x, const int pot_y);

    float* get_block_head(const int grid_x, const int grid_y, const int block_x,
                          const int block_y);
    float* get_grid_head(const int grid_x, const int grid_y);

    void resize_mag2_vec();

    int thresh_idx(const Eigen::Vector3i& pixel);
    int thresh_block_dim;
    int thresh_block_num_x;
    int thresh_block_num_y;
    std::unique_ptr<float[]> thresh_map;  // each block has a thresh_vec
    std::unique_ptr<float[]> thresh_map_backup;
    float weight[9] = {
        1.0f / 16, 2.0f / 16, 1.0f / 16,  //
        2.0f / 16, 4.0f / 16, 2.0f / 16,  //
        1.0f / 16, 2.0f / 16, 1.0f / 16,
    };  //

    // container for pixel selection, here to avoid allocate multiple times
    // resize only when we change the pot sizeS
    std::vector<float> pot_mag2_vec;
    std::vector<float> block_mag2_vec;
    std::vector<float> grid_mag2_vec;

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

    ImagePyramid::float_ptr
        mag2_ptr;  // unlikely with dso, we select only with mag2[0]
    std::vector<Eigen::Vector3i> candidates;

    int recursive_count;
};

//########################################################//
//#####################implementation#####################//

inline int PixelSelector::thresh_idx(const Eigen::Vector3i& pixel) {
    int x = (float)pixel(0) / thresh_block_dim;
    int y = (float)pixel(1) / thresh_block_dim;
    return idx(thresh_block_num_x, x, y);
}

inline float* PixelSelector::get_pot_head(const int grid_x, const int grid_y,
                                          const int block_x, const int block_y,
                                          const int pot_x, const int pot_y) {
    float* block_head = get_block_head(grid_x, grid_y, block_x, block_y);
    return &block_head[(pot_y * cam->width[0] + pot_x) * pot_dim];
}
inline float* PixelSelector::get_block_head(const int grid_x, const int grid_y,
                                            const int block_x,
                                            const int block_y) {
    float* grid_head = get_grid_head(grid_x, grid_y);
    return &grid_head[(block_y * cam->width[0] + block_x) * block_dim];
}
inline float* PixelSelector::get_grid_head(const int grid_x, const int grid_y) {
    return &mag2_ptr.get()[(grid_y * cam->width[0] + grid_x) * grid_dim];
}

}  // namespace mpl