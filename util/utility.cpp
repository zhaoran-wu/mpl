#include "utility.h"
#include <iostream>
#include <fstream>
#include <random>
#include <boost/format.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <glog/logging.h>
#include "as.h"
#include <yaml-cpp/yaml.h>


namespace util {

std::vector<std::string> getFilesInDir(const std::string& directory, const std::string& extension) {
    std::vector<std::string> files;
    auto paths = getFilePathsInDir(directory, extension);
    for (const auto& p : paths)
        files.push_back(p.string());
    return files;
}

std::vector<boost::filesystem::path> getFilePathsInDir(const std::string& directory, const std::string& extension) {
    std::vector<boost::filesystem::path> paths;
    boost::filesystem::directory_iterator it(directory);
    while (it != boost::filesystem::directory_iterator()) {
        if (is_regular_file(*it)) {
            if (extension != "-" && it->path().extension() == extension) {
                paths.push_back(it->path());
            } else if (extension == "-") {
                paths.push_back(it->path());
            }
        }
        it++;
    }
    std::sort(paths.begin(), paths.end(), [](const boost::filesystem::path& path1, const boost::filesystem::path& path2){ return path1.string() < path2.string(); });
    return paths;
}

uint64_t timestampFromFilename(const std::string& filename) {
    return timestampFromFilename(boost::filesystem::path(filename));
}

uint64_t timestampFromFilename(const boost::filesystem::path& path) {
    return std::stoull(path.stem().string());
}

Eigen::MatrixXf matrixFromTextFile(const std::string& filename) {
    std::vector<float> values;
    int rows = 0;
    int cols = 0;

    std::ifstream file;
    file.open(filename);
    if (file.is_open()) {
        float value;
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            while (iss >> value) {
                values.push_back(value);
            }
            rows++;
        }
        cols = values.size() / rows;
        file.close();
    } else {
        throw std::runtime_error("file could not be opened");
    }

    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> map(values.data(), rows, cols);
    return map;
}

std::string framesToChunkName(int frameStart, int frameEnd) {
    boost::format formatStart("%04d");
    formatStart % frameStart;
    boost::format formatEnd("%04d");
    formatEnd % frameEnd;
    return formatStart.str() + "_" + formatEnd.str();
}

std::string stampToFilename(uint64_t stamp) {
    boost::format format("%016d");
    format % stamp;
    return format.str();
}

std::pair<int, int> chunkNameToFrames(const std::string& filename) {
    auto pos = filename.find("_");
    std::string startStr = filename.substr(0, pos);
    std::string endStr = filename.substr(pos+1);
    return std::make_pair(std::stoi(startStr), std::stoi(endStr));
}

std::pair<int, int> chunkNameToFrames(const boost::filesystem::path& path) {
    return chunkNameToFrames(path.stem().string());
}

Eigen::Isometry3f interpolateDeltaPose(float lambda, const Eigen::Isometry3f& delta) {
    Eigen::Quaternionf qStart = Eigen::Quaternionf::Identity();
    Eigen::Quaternionf qEnd(delta.linear());
    Eigen::Quaternionf rot = qStart.slerp(lambda, qEnd);
    Eigen::Vector3f trans(lambda * delta.translation());
    Eigen::Isometry3f interpolated = Eigen::Translation3f(trans) * rot;
    return interpolated;
}

std::vector<int> loadLabels(const std::string &filename) {
    std::vector<int> labels;
    int label;
    std::ifstream file;
    file.open(filename);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            while (iss >> label) {
                labels.push_back(label);
            }
        }
        file.close();
    } else {
        throw std::runtime_error("file " + filename + " could not be opened");
    }
    return labels;
}

cv::Mat getDepthImage(std::vector<float>& depth, int cols) {
    return cv::Mat((int)depth.size()/cols, cols, CV_32FC1, depth.data());
}

void saveDepthImage(const std::string filename, std::vector<float> depth, int cols) {
    cv::Mat depthFloat((int)depth.size()/cols, cols, CV_32FC1, depth.data());
//    cv::patchNaNs(depth, 0); // set nans to scalar value

    cv::Mat image;
    depthFloat.convertTo(image, CV_16UC1, 1000); // convert to millimeters
    cv::imwrite(filename, image);
}

cv::Mat depthToImage(std::vector<float> depth, int cols, double depthThresh) {
    cv::Mat image((int)depth.size()/cols, cols, CV_32FC1, depth.data());

    // remove nans, threshold, convert uint8 and apply colormap
    cv::Mat dst;
    cv::patchNaNs(image, 0); // set nans to scalar value

    cv::Mat threshold;
    cv::threshold( image, threshold, depthThresh, depthThresh, cv::THRESH_TRUNC);

    cv::Mat img;
    threshold.convertTo(img, CV_8UC1, 255.0/depthThresh);
    cv::Mat mask = (img == cv::Mat(img.size(), CV_8UC1, cv::Scalar(0))); // get mask of nan pixels

    cv::Mat colored;
    cv::applyColorMap(img, colored, cv::COLORMAP_JET);
    colored.setTo(cv::Scalar(0, 0, 0), mask); // set nans to black
    return colored;
}

std::vector<float> floatsFromString(const std::string& str) {
    std::vector<float> vec;
    float val;
    std::stringstream ss(str);
    while (ss >> val) {
        vec.push_back(val);
    }
    return vec;
}

cv::Vec3b randomColor() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    return cv::Vec3b(dis(gen), dis(gen), dis(gen));
}

cv::Mat1i meshGrid(int rows, int cols, bool xCoords) {
    cv::Mat1i mesh;
    cv::Mat1i elem(xCoords ? 1 : rows, xCoords ? cols : 1);
    for (int i = 0; i < elem.total(); ++i) {
        elem.at<int>(i) = i;
    }
    cv::repeat(elem, xCoords ? rows : 1, xCoords ? 1 : cols, mesh);
    return mesh;
}

void cleanCreateDirectory(const boost::filesystem::path& path) {
    if (boost::filesystem::exists(path)) {
        boost::filesystem::remove_all(path);
    }
    boost::filesystem::create_directories(path);
}

void exists(const boost::filesystem::path& path, std::string msg) {
    bool exists = false;
    if (boost::filesystem::is_symlink(path)) {
        exists = boost::filesystem::exists(boost::filesystem::read_symlink(path));
    } else {
        exists = boost::filesystem::exists(path);
    }
    CHECK(exists) << "Path/file " << path.string() << " does not exist. " << msg;
}

SensorDataMap readSensorDataFromYaml(const std::string& filename) {
    SensorDataMap sensorDataMap;
    util::exists(filename, "config file");

    YAML::Node yaml = YAML::LoadFile(filename);
    for (const auto& n : yaml["sensors"]) {
        SensorData sensorData;

        std::string id = n["id"].as<std::string>();
        VLOG(1) << "reading config of sensor " << id;

        // general
        sensorData.id = id;
        if (YAML::Node disableNode = n["disable"]) {
            sensorData.disabled = disableNode.as<bool>();
        }

        // intrinsics
        auto dims = n["dimensions"].as<std::vector<int>>();
        sensorData.rows = dims[0];
        sensorData.cols = dims[1];
        auto intrinsics = n["intrinsics"].as<std::vector<float>>();
        sensorData.fx = intrinsics[0];
        sensorData.fy = intrinsics[1];
        sensorData.cx = intrinsics[2];
        sensorData.cy = intrinsics[3];

        // extrinsics
        if (YAML::Node extNode = n["extrinsics"]) {
            VLOG(1) << "setting extrinsics";
            int i = 0;
            auto extrinsicsParams = extNode.as<std::vector<float>>();
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 4; ++c) {
                    sensorData.extrinsics(r, c) = extrinsicsParams[i++];
                }
            }
        }

        sensorDataMap[id] = sensorData;
    }

    return sensorDataMap;
}

void createCameraFrames(const trajectory_io::Trajectory& trajectory, int rows, int cols, double focal, Eigen::MatrixXd& P1, Eigen::MatrixXd& P2) {
    int numPoses = (int)trajectory.size();
    P1.resize(numPoses*8, 3);
    P2.resize(numPoses*8, 3);

    double focalLengthVisu = 0.25;
    double widthHalfVisu = (double)cols/2 * focalLengthVisu / focal;
    double heightHalfVisu = (double)rows/2 * focalLengthVisu / focal;

    Eigen::Vector3d topLeftLocal(-widthHalfVisu,-heightHalfVisu,focalLengthVisu);
    Eigen::Vector3d topRightLocal(widthHalfVisu,-heightHalfVisu,focalLengthVisu);
    Eigen::Vector3d bottomLeftLocal(-widthHalfVisu,heightHalfVisu,focalLengthVisu);
    Eigen::Vector3d bottomRightLocal(widthHalfVisu,heightHalfVisu,focalLengthVisu);

    for (int camIdx = 0; camIdx < numPoses; ++camIdx) {
        Eigen::Isometry3d pose = trajectory.atIndex(camIdx).cast<double>();
        Eigen::RowVector3d origin = pose.translation().transpose();
        Eigen::RowVector3d topleft = (pose*topLeftLocal).transpose();
        Eigen::RowVector3d topright = (pose*topRightLocal).transpose();
        Eigen::RowVector3d bottomleft = (pose*bottomLeftLocal).transpose();
        Eigen::RowVector3d bottomright = (pose*bottomRightLocal).transpose();

        P1.row(8*camIdx+0) = origin;
        P1.row(8*camIdx+1) = origin;
        P1.row(8*camIdx+2) = origin;
        P1.row(8*camIdx+3) = origin;

        P2.row(8*camIdx+0) = topleft;
        P2.row(8*camIdx+1) = bottomleft;
        P2.row(8*camIdx+2) = bottomright;
        P2.row(8*camIdx+3) = topright;

        P1.row(8*camIdx+4) = topleft;
        P1.row(8*camIdx+5) = bottomleft;
        P1.row(8*camIdx+6) = bottomright;
        P1.row(8*camIdx+7) = topright;

        P2.row(8*camIdx+4) = bottomleft;
        P2.row(8*camIdx+5) = bottomright;
        P2.row(8*camIdx+6) = topright;
        P2.row(8*camIdx+7) = topleft;
    }
}

} // namespace util