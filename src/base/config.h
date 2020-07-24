#pragma once
#include <mutex>
#include <string>

namespace mpl {
class Config {
   public:
    /**
     * @brief Get the Instance object, it define a static object of Config class
     *
     * @return the reference of the unique obeject
     */
    static Config& getInstance();

    bool readYamlFile(const std::string& file_path);
    // make sure delete those two to keep the rule of singelton
    Config(const Config&) = delete;
    Config& operator=(const Config&) = delete;

   private:
    // those two are set as private --> no chance to create a non-static Config
    // object
    Config() = default;
    ~Config() = default;

   public:
    int PYRAMID_LVLS;
    int THRESHOLD_BLOCK_DIM;
    int PIXEL_SELECTION_POT_DIM;
    int PIXEL_SELECTION_HERURISTIC_CONST;
    int PIXEL_SELECTION_NUM;
    float PIXEL_SELECTION_DOWNWEIGHT;
    int PIXEL_SELECTION_RECURSIVE_LIMIT;

    float OPTIMIZATION_TRANS_SCALE;
    float OPTIMIZATION_ROTATION_SCALE;
    float OPTIMIZATION_ALPHA_SCALE;
    float OPTIMIZATION_BETA_SCALE;
    float OPTIMIZATION_IDEPTH_SCALE;
    float OPTIMIZATION_STEP_MIN;
    float OPTIMIZATION_LAMDA_INIT;
    float OPTIMIZATION_LAMDA_MAX;
    float OPTIMIZATION_LAMDA_MIN;
    float OPTIMIZATION_LAMDA_FAILED_PENALIZE;
    float OPTIMIZATION_LAMDA_SUCCESS_PENALIZE;

    // debug
    std::mutex debug_distance_map_mutex;
    bool DEBUG_DISTANCE_MAP = false;

    std::mutex debug_key_frame_synetci_img_alignment_mutex;
    bool DEBUG_KEY_FRAME_SYNETIC_IMAGE_ALIGNMENT = false;

    std::mutex debug_pixel_selection_mutex;
    bool DEBUG_PIXEL_SELECTION = false;

    std::mutex debug_image_pyramid_mutex;
    bool DEBUG_IMAGE_PYRAMID = false;

    std::mutex debug_coarse_to_fine_tracking_mutex;
    bool DEBUG_COARSE_TO_FINE_TRACKING = false;

    // std::mutex debug_pixel_selection_mutex;
    // bool DEBUG_PIXEL_SELECTION = false;

    const int max_iteration_each_lvl[5] = {50, 30, 25, 20, 10};
    const float lamda_init_each_lvl[5] = {1e-8, 1e-5, 1e-1, 1, 1};  // very sensitive : e.g motion blur
    const float lamda_min_eahc_lvl[5] = {1e-14, 1e-10, 1e-9, 1e-7, 1e-7};
    const int huber_residual_each_lvl[5] = {40, 55, 75, 100, 120};

    static const int WINDOW_SIZE = 7;
};

}  // namespace mpl