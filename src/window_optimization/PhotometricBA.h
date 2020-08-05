#pragma once

#include <unordered_set>
#include <vector>

#include "ceres/ceres.h"

#include "bundle_adjustment.h"
#include "candidate_manager.h"
#include "ceres/PhotometricBAConfig.h"

namespace mpl {
class Frame;

class PhotometricResidual;
class PhotometricBAEvaluationCallback;
class PhotometricBAIterationCallback;

class PhotometricBA {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    friend class PhotometricResidual;
    friend class PhotometricCostFunction;
    friend class PhotometricBAIterationCallback;

    PhotometricBA(const PhotometricBAConfig& config = PhotometricBAConfig());
    ~PhotometricBA();

    // main function to solve and merge optimization result
    void solve(CandidateManager& cm);

   private:
    void reset();

    void prepareOptimization(CandidateManager& cm, BundleAdjustment& problem);

    void mergeOptimization(CandidateManager& cm, std::vector<PhotometricResidual*>& obsToRemove) const;

    void removeBadObservations(const std::vector<PhotometricResidual*>& obsToRemove) const;

    void freeFixedKeyframesMemory();

   private:
    // optimization options
    PhotometricBAConfig options;

    // iteration callback to control thresholds and termination criteria
    std::unique_ptr<PhotometricBAIterationCallback> iterCallback;

    // statistics
    int numFrames;
    int numPoints;
    int numResiduals;

    // actual optimization data
    std::vector<Frame::ptr> activeKeyframes;

    std::vector<PhotometricResidual*> activeObservations;

    std::unordered_map<Frame::ptr, std::vector<Candidate>>* candidate_map_ptr;
};
}  // namespace mpl