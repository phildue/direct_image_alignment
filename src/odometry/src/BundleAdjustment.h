#ifndef VSLAM_BUNDLE_ADJUSTMENT_H__
#define VSLAM_BUNDLE_ADJUSTMENT_H__
#include <ceres/ceres.h>
#include <vector>
#include "core/core.h"

namespace pd::vslam::mapping{
        class BundleAdjustment
        {
                public:
                
                BundleAdjustment(const std::vector<FrameRgb::ConstShPtr>& frames, const std::vector<Point3D::ConstShPtr>& points);
                
                void optimize();

                void update(const std::vector<FrameRgb::ShPtr>& frames) const;
                void update(const std::vector<Point3D::ShPtr>& points) const;

                double computeError() const;

                private:
                ceres::Problem _problem;
                std::map<std::uint64_t,SE3d> _poses;
                std::map<std::uint64_t,Vec3d> _points;
                const std::vector<FrameRgb::ConstShPtr> _frames;

        };
}

#endif