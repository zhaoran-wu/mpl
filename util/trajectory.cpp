#include "trajectory.h"

#include <fstream>
#include <iostream>

namespace trajectory_io {

void Trajectory::read(const std::string& filename, const Format& format) {
    std::ifstream file;
    file.open(filename);
    if (file.is_open()) {
        std::string line;
        int64_t t;
        float tx, ty, tz, roll, pitch, yaw;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            while (iss >> t) {
                Eigen::Isometry3f pose;
                switch (format) {
                case FORMAT_RPY: {
                    iss >> tx;
                    iss >> ty;
                    iss >> tz;
                    iss >> roll;
                    iss >> pitch;
                    iss >> yaw;
                    pose = Eigen::Translation3f(Eigen::Vector3f(tx, ty, tz)) *
                           Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()) *
                           Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()) *
                           Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX());
                    break;
                }
                case FORMAT_MAT: {
                    pose.setIdentity();
                    for (int r = 0; r < 3; ++r) {
                        for (int c = 0; c < 4; ++c) {
                            iss >> pose.matrix()(r, c);
                        }
                    }
                    break;
                }
                default:
                    throw std::runtime_error("specified format does not exist");
                }
                trajectory_.insert(std::make_pair(t, pose));
            }
        }
        file.close();
    } else {
        throw std::runtime_error("could not open file " + filename);
    }
}

void Trajectory::write(const std::string& filename, const Format& format) const {
    std::ofstream file;
    file.open(filename, std::ios::out);
    if (file.is_open()) {
        for (const auto& t : trajectory_) {
            switch (format) {
            case FORMAT_RPY: {
                auto trans = t.second.translation();
                auto rot = t.second.linear().eulerAngles(2, 1, 0);
                file << t.first << ' ' << trans(0) << ' ' << trans(1) << ' ' << trans(2) << ' ' << rot(2) << ' '
                     << rot(1) << ' ' << rot(0) << '\n';
                break;
            }
            case FORMAT_MAT: {
                file << t.first;
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 4; ++c) {
                        file << ' ' << t.second.matrix()(r, c);
                    }
                }
                file << '\n';
                break;
            }
            default:
                throw std::runtime_error("specified format does not exist");
            }
        }
        file.close();
    } else {
        throw std::runtime_error("could not open file " + filename);
    }
}

void Trajectory::insert(const Eigen::Isometry3f& pose, int64_t stamp) {
    trajectory_.insert(std::make_pair(stamp, pose));
}

void Trajectory::clear() {
    trajectory_.clear();
}

int Trajectory::size() const {
    return (int)trajectory_.size();
}

Eigen::Isometry3f Trajectory::interpolate(int64_t stamp, bool* extrapolated) const {
    if (extrapolated) {
        *extrapolated = false;
    }
    TrajectorySet::const_iterator itHigh = trajectory_.upper_bound(std::make_pair(stamp, Eigen::Isometry3f::Identity()));
    TrajectorySet::const_iterator itLow = itHigh; // itHigh points to next element after stamp, so itLow must be one before itHigh
    if (itHigh == trajectory_.cbegin()) { // special case, we need to extrapolate at beginning
        std::advance(itHigh, 1);
        if (extrapolated) {
            *extrapolated = true;
        }
    } else {
        std::advance(itLow, -1);
    }

    if (itLow->first == stamp) { // check for exact match
        return itLow->second;
    }

    if (itHigh == trajectory_.cend()) { // extrapolate at end
        std::advance(itLow, -1);
        std::advance(itHigh, -1);
        if (extrapolated) {
            *extrapolated = true;
        }
    }

    int64_t stamp1 = itLow->first;
    int64_t stamp2 = itHigh->first;
    float delta = stamp > stamp1 ? (float)(stamp - stamp1) : -(float)(stamp1 - stamp); // handle special case because we use unsigned timestamps
    float lambda = delta / (float)(stamp2 - stamp1);
    return interpolate(itLow->second, itHigh->second, lambda);
}

Eigen::Isometry3f Trajectory::interpolate(const Eigen::Isometry3f& p1, const Eigen::Isometry3f& p2, float lambda) const {
    Eigen::Quaternionf q1(p1.linear());
    Eigen::Quaternionf q2(p2.linear());
    Eigen::Quaternionf rot = q1.slerp(lambda, q2);

    Eigen::Translation3f trans(lambda * p2.translation() + (1 - lambda) * p1.translation());

    return trans * rot;
}

const Eigen::Isometry3f& Trajectory::atStamp(uint64_t stamp) const {
    return trajectory_.find(dummyPose(stamp))->second;
}

int Trajectory::getStampIndex(uint64_t stamp) const {
    TrajectorySet::const_iterator it = trajectory_.find(dummyPose(stamp));

    if (it != trajectory_.end())
        return std::distance(trajectory_.begin(), it);
    else
        throw std::runtime_error("no pose for given stamp " + std::to_string(stamp) + " in trajectory");
}

uint64_t Trajectory::getIndexStamp(int index) const {
    if (index >= 0 && index < size()) {
        TrajectorySet::const_iterator it = trajectory_.cbegin();
        std::advance(it, index);
        return it->first;
    } else {
        throw std::runtime_error("requesting invalid index " + std::to_string(index) + " from trajectory");
    }
}

const Eigen::Isometry3f& Trajectory::atIndex(int index) const {
    if (index >= 0 && index < size()) {
        TrajectorySet::const_iterator it = trajectory_.cbegin();
        std::advance(it, index);
        return it->second;
    } else {
        throw std::runtime_error("requesting invalid index " + std::to_string(index) + " from trajectory");
    }
}

Trajectory::PoseStamped Trajectory::dummyPose(const uint64_t stamp) const {
    return std::make_pair(stamp, Eigen::Isometry3f::Identity());
}

void Trajectory::replaceAtStamp(const uint64_t stamp, const Eigen::Isometry3f& pose) {
    TrajectorySet::const_iterator it = trajectory_.find(dummyPose(stamp));
    if (it != trajectory_.end()) {
        it = trajectory_.erase(it);
        trajectory_.insert(it, std::make_pair(stamp, pose));
    } else {
        throw std::runtime_error("no pose for given stamp " + std::to_string(stamp) + " in trajectory");
    }
}

void Trajectory::replaceAtIndex(const int index, const Eigen::Isometry3f& pose) {
    uint64_t stamp = getIndexStamp(index);
    replaceAtStamp(stamp, pose);
}

bool Trajectory::canInterpolate(const uint64_t stamp) const {
    const uint64_t lower = trajectory_.cbegin()->first;
    const uint64_t upper = trajectory_.crbegin()->first;
    return stamp >= lower && stamp <= upper;
}

} // namespace trajectory_io