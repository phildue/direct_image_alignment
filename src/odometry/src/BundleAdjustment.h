#ifndef VSLAM_BUNDLE_ADJUSTMENT_H__
#define VSLAM_BUNDLE_ADJUSTMENT_H__
#include <ceres/ceres.h>
#include <vector>
#include "core/core.h"

namespace pd::vslam::mapping{
        class BundleAdjustment
        {
                public:
                typedef std::shared_ptr<BundleAdjustment> ShPtr;
                typedef std::unique_ptr<BundleAdjustment> UnPtr;
                typedef std::shared_ptr<const BundleAdjustment> ConstShPtr;
                typedef std::unique_ptr<const BundleAdjustment> ConstUnPtr;

                BundleAdjustment();
                void optimize();
                void optimize(const std::vector<FrameRgbd::ShPtr>& frames, const std::vector<Point3D::ShPtr>& points);
                
                void insertFrame(std::uint64_t frameId, const SE3d& pose, const Mat3d& K);
                void insertPoint(std::uint64_t pointId, const Vec3d& point);
                void insertObservation(std::uint64_t pointId, std::uint64_t frameId, const Vec2d& observation);
                void insertFrame(FrameRgb::ConstShPtr frame);
                void insertPoint(Point3D::ConstShPtr point);
                
                template<class Iterator>
                void insertFrames(Iterator first, Iterator last)
                { std::for_each(first,last,[&](FrameRgb::ConstShPtr f){ insertFrame(f);});}
                
                template<class Iterator>
                void insertPoints(Iterator first, Iterator last)
                { std::for_each(first,last,[&](Point3D::ConstShPtr p){ insertPoint(p);});}


                PoseWithCovariance::UnPtr getPose(std::uint64_t frameId) const;
                Vec3d getPosition(std::uint64_t pointId) const;
                
                template<class Iterator>
                void getPoses(Iterator first, Iterator last) const
                { std::for_each(first,last,[&](const FrameRgb::ShPtr& f){ f->set(*getPose(f->id()));});}
                
                template<class Iterator>
                void getPositions(Iterator first, Iterator last) const
                { std::for_each(first,last,[&](const Point3D::ShPtr& p){ p->position() = getPosition(p->id());});}


                double computeError() const;

                private:
                ceres::Problem _problem;
                std::map<std::uint64_t,SE3d> _poses;
                std::map<std::uint64_t,Mat3d> _Ks;
                std::map<std::uint64_t,Vec3d> _points;
                std::vector<FrameRgb::ConstShPtr> _frames;

        };
}

#endif