#ifndef VSLAM_RGBD_ODOMETRY_KALMAN_H__
#define VSLAM_RGBD_ODOMETRY_KALMAN_H__
#include "core/core.h"
#include "kalman/kalman.h"
#include "RgbdOdometry.h"
namespace pd::vision{
class RgbdOdometryKalman {
        public:
        typedef std::shared_ptr<RgbdOdometryKalman> ShPtr;
        typedef std::shared_ptr<const RgbdOdometryKalman> ConstPtr;

        RgbdOdometryKalman(Camera::ShPtr camera, double minGradient = 50, int nLevels = 4, int maxIterations = 20, double convergenceThreshold = 1e-4, double dampingFactor = 1.0);

        SE3d estimate(const Image& fromRgb,const DepthMap& fromDepth, const Image& toRgb, std::uint64_t t) const;
        private:
        const KalmanFilterSE3::UnPtr _filter;
        const RgbdOdometry::UnPtr _aligner;
        

};
}
#endif// VSLAM_RGBD_ODOMETRY

