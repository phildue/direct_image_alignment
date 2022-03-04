#ifndef VSLAM_RGBD_ODOMETRY
#define VSLAM_RGBD_ODOMETRY
#include "core/core.h"
#include "lukas_kanade/lukas_kanade.h"
#include "solver/solver.h"
namespace pd::vision{
class RgbdOdometry{
        public:
        typedef std::shared_ptr<RgbdOdometry> ShPtr;
        typedef std::unique_ptr<RgbdOdometry> UnPtr;
        typedef std::shared_ptr<const RgbdOdometry> ConstShPtr;
        typedef std::unique_ptr<const RgbdOdometry> ConstUnPtr;

        RgbdOdometry(double minGradient = 0, int nLevels = 4, int maxIterations = 30, double convergenceThreshold = 1e-9, double dampingFactor = 1.0);

        PoseWithCovariance::UnPtr align(FrameRgbd::ConstShPtr from, FrameRgb::ConstShPtr to) const;
        PoseWithCovariance::UnPtr align(std::vector<FrameRgbd::ConstShPtr>& from,  FrameRgb::ConstShPtr to) const;

        protected:
        const int _nLevels;
        const int _maxIterations;
        const double _minGradient;
        const double _convergenceThreshold;
        const double _dampingFactor;
        const Loss::ShPtr _loss;
        

};
}
#endif// VSLAM_RGBD_ODOMETRY

