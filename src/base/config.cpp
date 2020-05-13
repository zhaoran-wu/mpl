#include "config.h"
#include "../util/utility.h"
#include <string>
#include <yaml-cpp/yaml.h>
namespace mpl {

#define read(name, type) this->name = global_node[#name].as<type>();

bool Config::readYamlFile(const std::string& file_path) {
    util::exists(file_path);
    YAML::Node global_node = YAML::LoadFile(file_path);

    read(PYRAMID_LVLS, int);
    read(THRESHOLD_BLOCK_DIM, int);
    read(PIXEL_SELECTION_POT_DIM, int);
    read(PIXEL_SELECTION_HERURISTIC_CONST, int);
    read(PIXEL_SELECTION_NUM, int);
    read(PIXEL_SELECTION_DOWNWEIGHT, float);
    read(PIXEL_SELECTION_RECURSIVE_LIMIT, int);
    read(OPTIMIZATION_TRANS_SCALE, float);
    read(OPTIMIZATION_ROTATION_SCALE, float);
    read(OPTIMIZATION_ALPHA_SCALE, float);
    read(OPTIMIZATION_BETA_SCALE, float);
    read(OPTIMIZATION_LAMDA_INIT, float);
    read(OPTIMIZATION_LAMDA_MAX, float);
    read(OPTIMIZATION_LAMDA_MIN, float);
    read(OPTIMIZATION_LAMDA_FAILED_PENALIZE, float);
    read(OPTIMIZATION_LAMDA_SUCCESS_PENALIZE, float);

    return true;
}

Config& Config::getInstance() {
    static Config unique_config;
    return unique_config;
}

}  // namespace mpl