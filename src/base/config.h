#pragma once
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
};

}  // namespace mpl