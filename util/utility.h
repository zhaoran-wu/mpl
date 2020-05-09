#pragma once

#include "types.h"
#include <vector>
#include <string>
#include <utility>
#include <Eigen/Dense>
#include <boost/filesystem.hpp>
#include <opencv2/core/core.hpp>
#include "trajectory.h"


namespace util {

enum Label {
    UNLABELED = 0,
    EGO_VEHICLE = 1,
    RECTIFICATION_BORDER = 2,
    OUT_OF_ROI = 3,
    STATIC = 4,
    DYNAMIC = 5,
    GROUND = 6,
    ROAD = 7,
    SIDEWALK = 8,
    PARKING = 9,
    RAIL_TRACK = 10,
    BUILDING = 11,
    WALL = 12,
    FENCE = 13,
    GUARD_RAIL = 14,
    BRIDGE = 15,
    TUNNEL = 16,
    POLE = 17,
    POLEGROUP = 18,
    TRAFFIC_LIGHT = 19,
    TRAFFIC_SIGN = 20,
    VEGETATION = 21,
    TERRAIN = 22,
    SKY = 23,
    PERSON = 24,
    RIDER = 25,
    CAR = 26,
    TRUCK = 27,
    BUS = 28,
    CARAVAN = 29,
    TRAILER = 30,
    TRAIN = 31,
    MOTORCYCLE = 32,
    BICYCLE = 33,
    LICENSE_PLATE = -1,
};

// cityscape label colors as defined in https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
// note: opencv is bgr, python code has rgb
static std::vector<cv::Vec3b> labelColors = {cv::Vec3b(0, 0, 0),     // unlabeled
                                             cv::Vec3b(0, 0, 0),     // ego vehicle
                                             cv::Vec3b(0, 0, 0),     // rectification border
                                             cv::Vec3b(0, 0, 0),     // out of roi
                                             cv::Vec3b(0, 0, 0),     // static
                                             cv::Vec3b(0, 74, 111),  // dynamic
                                             cv::Vec3b(81, 0, 81),   // ground
                                             cv::Vec3b(128, 64, 128),        // road
                                             cv::Vec3b(232, 35, 244),        // sidewalk
                                             cv::Vec3b(160, 170, 250),       // parking
                                             cv::Vec3b(140, 150, 230),       // rail track
                                             cv::Vec3b(70, 70, 70),  // building
                                             cv::Vec3b(156, 102, 102),       // wall
                                             cv::Vec3b(153, 153, 190),       // fence
                                             cv::Vec3b(180, 165, 180),       // guard rail
                                             cv::Vec3b(100, 100, 150),       // bridge
                                             cv::Vec3b(90, 120, 150),        // tunnel
                                             cv::Vec3b(153, 153, 153),       // pole
                                             cv::Vec3b(153, 153, 153),       // polegroup
                                             cv::Vec3b(30, 170, 250),        // traffic light
                                             cv::Vec3b(0, 220, 220),         // traffic sign
                                             cv::Vec3b(35, 142, 107),        // vegetation
                                             cv::Vec3b(152, 251, 152),       // terrain
                                             cv::Vec3b(180, 130, 70),        // sky
                                             cv::Vec3b(60, 20, 220),         // person
                                             cv::Vec3b(0, 0, 255),   // rider
                                             cv::Vec3b(142, 0, 0),   // car
                                             cv::Vec3b(70, 0, 0),    // truck
                                             cv::Vec3b(100, 60, 0),  // bus
                                             cv::Vec3b(90, 0, 0),    // caravan
                                             cv::Vec3b(110, 0, 0),   // trailer
                                             cv::Vec3b(100, 80, 0),  // train
                                             cv::Vec3b(230, 0, 0),   // motorcycle
                                             cv::Vec3b(32, 11, 119),         // bicycle
                                             cv::Vec3b(142, 0, 0),   // license plate
};

// colors from matplotlib
enum MplColors {
    BLUE = 0,
    ORANGE = 1,
    GREEN = 2,
    RED = 3,
    PURPLE = 4
};

static std::vector<cv::Vec3b> mplColors = {cv::Vec3b(31,119,180),
                                           cv::Vec3b(255,127,14),
                                           cv::Vec3b(44,160,44),
                                           cv::Vec3b(214,39,40),
                                           cv::Vec3b(148,103,189)
};

/**
 * \brief return all files with specified extension in a directory sorted alphabetically
 * @param directory search directory
 * @param extension file extension such as .txt for textfiles
 * @return vector of strings containing full path to all files that match
 */
std::vector<std::string> getFilesInDir(const std::string& directory, const std::string& extension = "");

/**
 * \brief eturn all files with specified extension in a directory sorted alphabetically
 * @param directory search directory
 * @param extension file extension such as .txt for textfiles
 * @return vector of boost paths containing full path to all files that match
 */
std::vector<boost::filesystem::path> getFilePathsInDir(const std::string& directory, const std::string& extension = "");

/**
 * \brief convert filename (16 digit string) to timestamp
 * @param filename filename
 * @return timestamp
 */
uint64_t timestampFromFilename(const std::string& filename);

/**
 * \brief convert path to file with filename (16 digit string) to timestamp
 * @param path to file
 * @return timestamp
 */
uint64_t timestampFromFilename(const boost::filesystem::path& path);

/**
 * \brief read matrix values from text file
 * @param filename text file containing values
 * @return matrix
 */
Eigen::MatrixXf matrixFromTextFile(const std::string& filename);

/**
 * \brief convert start and end frame indices to chunck filename without extension
 * @param frameStart first frame of chunk
 * @param frameEnd last frame of chunk
 * @return chunck filename as string
 */
std::string framesToChunkName(int frameStart, int frameEnd);

/**
 * \brief convert timestamp in microseconds to 16 digit filename without extension
 * @param stamp timestamp
 * @return string of timestamp
 */
std::string stampToFilename(uint64_t stamp);

/**
 * \brief convert file name of chunck to start and end frame indices
 * @param filename filename
 * @return pair containing first and last frame contained in chunk
 */
std::pair<int, int> chunkNameToFrames(const std::string& filename);

std::pair<int, int> chunkNameToFrames(const boost::filesystem::path& path);

/**
 * \brief linear interpolation of isometry
 * @param lambda running variable [0,1]
 * @param delta delta pose as isometry
 * @return interpolated pose
 */
Eigen::Isometry3f interpolateDeltaPose(float lambda, const Eigen::Isometry3f& delta);

/**
 * \brief load white space separated integer values from text file
 * @param filename path to file to read
 * @return vector of integers
 */
std::vector<int> loadLabels(const std::string &filename);

/**
 * \brief convert vector of values into matrix
 * @param depth depth values in row major order
 * @param cols number of columns of the output matrix
 * @return depths in matrix form
 */
cv::Mat getDepthImage(std::vector<float>& depth, int cols);

/**
 * \brief save depths as image file
 * @param filename path of image to write
 * @param depth depth values in row major order as vector
 * @param cols number of columns of output image
 */
void saveDepthImage(const std::string filename, std::vector<float> depth, int cols);

/**
 * \brief create color coded depth image for visualization
 * @param depth depth depth values in row major order as vector
 * @param cols number of columns of image
 * @param depthThresh maximum distance to show
 * @return color image
 */
cv::Mat depthToImage(std::vector<float> depth, int cols, double depthThresh = 20.0);

/**
 * \brief convert a string of floats into a vector of floats
 * @param str string with float values separated by white spaces
 * @return vector of floats
 */
std::vector<float> floatsFromString(const std::string& str);

/**
 * \brief create random color
 * @return random color
 */
cv::Vec3b randomColor();

/**
 * \brief create a matrix holding x or y coordinate of each pixel as pixel value
 * @param rows number of rows
 * @param cols number of columns
 * @param xCoords if true, pixels hold x coordinates, false for y coordinates
 * @return matrix holding coordinates
 */
cv::Mat1i meshGrid(int rows, int cols, bool xCoords);

/**
 * \brief create directory if it does not exist jet or clean it if it exists
 * @param path path to directory
 */
void cleanCreateDirectory(const boost::filesystem::path& path);

/**
 * \brief checks if path or file exists
 * @param path path or file
 * @param additional message to print
 */
void exists(const boost::filesystem::path& path, std::string msg = "");

/**
 * \brief write eigen matrix to text file which can be read by numpy
 * @param filename file name
 * @param mat matrix to write to file
 */
template <typename T>
void saveEigenMatrix(const std::string& filename, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& mat);

template <typename T>
void saveCVMatrix(const std::string& filename, const cv::Mat& mat);

/**
 * \brief reads sensor model parameters from yaml config file
 * @param filename yaml file
 * @return map from sensor id to sensordata struct
 */
std::map<std::string, SensorData> readSensorDataFromYaml(const std::string& filename);

void createCameraFrames(const trajectory_io::Trajectory& trajectory, int rows, int cols, double focal, Eigen::MatrixXd& P1, Eigen::MatrixXd& P2);

//template <typename DerivedV, typename DerivedF, typename DerivedUV>
//bool writePLY(
//        const std::string & filename,
//        const Eigen::MatrixBase<DerivedV> & V,
//        const Eigen::MatrixBase<DerivedF> & F,
//        const Eigen::MatrixBase<DerivedUV> & UV_V,
//        const bool ascii);

} // namespace util

#include "utility.hpp"