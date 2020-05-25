#include "cam_data.h"
#include "../util/utility.h"
#include "config.h"
#include <string>
#include <yaml-cpp/yaml.h>
namespace mpl {

#define STRINGFY(name) #name

bool CamData::readYamlFile(const std::string& file_path) {
    util::exists(file_path);
    YAML::Node global_node = YAML::LoadFile(file_path);

    auto& config = Config::getInstance();
    lvls = config.PYRAMID_LVLS;

    std::vector<float> intrinsic =
        global_node["intrinsics"].as<std::vector<float>>();
    std::vector<int> size = global_node["dimensions"].as<std::vector<int>>();

    for (int lvl = 0; lvl < lvls; ++lvl) {
        float fx_lvl = intrinsic[0] / pow(2, lvl);
        float fy_lvl = intrinsic[1] / pow(2, lvl);
        float cx_lvl = (intrinsic[2] + 0.5f) / pow(2, lvl) - 0.5f;
        float cy_lvl = (intrinsic[3] + 0.5f) / pow(2, lvl) - 0.5f;

        fx.push_back(fx_lvl);
        fy.push_back(fy_lvl);
        cx.push_back(cx_lvl);
        cy.push_back(cy_lvl);

        Eigen::Matrix3f K_lvl;
        K_lvl << fx_lvl, 0.0f, cx_lvl, 0.0f, fy_lvl, cy_lvl, 0.0f, 0.0f, 1.0f;
        K.push_back(K_lvl);
        Eigen::Matrix3f K_inv_lvl = K_lvl.inverse();
        K_inv.push_back(K_inv_lvl);

        int h_lvl = size[0] >> lvl;
        int w_lvl = size[1] >> lvl;

        height.push_back(h_lvl);
        width.push_back(w_lvl);
    }

    return true;
}

}  // namespace mpl