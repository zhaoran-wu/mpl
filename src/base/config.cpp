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
    return true;
}

Config& Config::getInstance() {
    static Config unique_config;
    return unique_config;
}

}  // namespace mpl