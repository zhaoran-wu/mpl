#pragma once

#include <utility>
#include <set>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

namespace trajectory_io {

/**
 * \brief class representing a trajectory which consists of timely ordered pairs of timestamps and poses
 */
class Trajectory {
    struct compare;

    typedef std::pair<int64_t, Eigen::Isometry3f> PoseStamped;
    typedef std::set< PoseStamped, compare, Eigen::aligned_allocator<PoseStamped> > TrajectorySet;

    // compare timestamps only
    struct compare {
        bool operator() (const PoseStamped& lhs, const PoseStamped& rhs) const {
            return lhs.first < rhs.first;
        }
    };

    TrajectorySet trajectory_;

public:
    enum Format { FORMAT_RPY, FORMAT_MAT };

    /**
     * \brief access element by index and check for ranges
     * @param index index to access
     * @return pose at given index
     */
    const Eigen::Isometry3f& atIndex(int index) const;

    /**
     * \brief get pose at given stamp (no interpolation)
     * @param stamp
     * @return pose at given stamp
     */
    const Eigen::Isometry3f& atStamp(uint64_t stamp) const;

    /**
     * \brief read trajectory from text file
     * @param filename filename (full path)
     * @param format trajectory format (rotation matrix or euler angles)
     */
    void read(const std::string& filename, const Format& format = FORMAT_MAT);

    /**
     * \brief write trajectory to text file
     * @param filename filename (full path)
     * @param format trajectory format (rotation matrix or euler angles)
     */
    void write(const std::string& filename, const Format& format = FORMAT_MAT) const;

    /**
     * \brief insert new pose to trajectory
     * @param pose pose
     * @param stamp timestamp of pose
     */
    void insert(const Eigen::Isometry3f& pose, int64_t stamp);

    /**
     * \brief remove all poses from trajectory
     */
    void clear();

    /**
     * \brief size accessor
     * @return number of poses in trajectory
     */
    int size() const;

    /**
     * \brief interpolate pose at given timestamp
     * @param stamp timestamp
     * @param extrapolated will be set to true if function needs to extrapolate for timestamp
     * @return interpolated pose
     */
    Eigen::Isometry3f interpolate(int64_t stamp, bool* extrapolated = nullptr) const;

    /**
     * \brief get index at given stamp (no interpolation)
     * @param stamp
     * @return index of given stamp
     */
    int getStampIndex(uint64_t stamp) const;

    /**
     *
     * @param index
     * @return timestamp for given index
     */
    uint64_t getIndexStamp(int index) const;

    /**
     * \brief modify pose at given index
     * @param index index at which we want to replace pose
     * @param pose new pose
     */
    void replaceAtIndex(const int index, const Eigen::Isometry3f& pose);

    /**
     * \brief modify pose at given timestamp
     * @param stamp timestamp at which we want to replace pose
     * @param pose new pose
     */
    void replaceAtStamp(const uint64_t stamp, const Eigen::Isometry3f& pose);

    /**
     * \brief checks if given timestamp lies within trajectory
     * @param stamp time stamp we want to check
     * @return true if time stamp is within trajectory
     */
    bool canInterpolate(const uint64_t stamp) const;

private:
     /**
     * \brief linear interpolation os two poses (linear in translation and rotation)
     * @param p1 first pose
     * @param p2 second pose
     * @param lambda interpolation parameter [0..1]
     * @return interpolated pose
     */
     Eigen::Isometry3f interpolate(const Eigen::Isometry3f& p1, const Eigen::Isometry3f& p2, float lambda) const;

    /**
     * \brief create a dummy pose used for searching for a time stamp
     * @param random pose with stamp
     * @return pose with given time stamp
     */
    PoseStamped dummyPose(const uint64_t stamp) const;
};

} // namespace trajectory_io