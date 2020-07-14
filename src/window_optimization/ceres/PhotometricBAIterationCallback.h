
#pragma once

#include "PhotometricBA.h"

namespace mpl {
class PhotometricBAIterationCallback : public ceres::IterationCallback {
   public:
    PhotometricBAIterationCallback(const PhotometricBA& bundleAdjustment);
    ~PhotometricBAIterationCallback();

    ceres::CallbackReturnType operator()(
        const ceres::IterationSummary& summary);

   private:
    void backup() const;
    bool checkTerminationCriteria() const;

   private:
    const PhotometricBA& bundle;
};
}  // namespace mpl