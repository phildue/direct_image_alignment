#ifndef VSLAM_RGBD_ODOMETRY
#define VSLAM_RGBD_ODOMETRY
#include "core/types.h"
#include "core/Camera.h"

namespace pd::vision{
class RgbdOdometry{
        public:
        typedef std::shared_ptr<RgbdOdometry> ShPtr;
        typedef std::shared_ptr<const RgbdOdometry> ConstPtr;

        RgbdOdometry(Camera::ShPtr camera, double minGradient = 50, int nLevels = 4, int maxIterations = 20, double convergenceThreshold = 1e-4, double dampingFactor = 1.0);

        SE3d estimate(const Image& fromRgb,const DepthMap& fromDepth, const Image& toRgb);
        private:
        const Camera::ShPtr _camera;
        const int _nLevels;
        const int _maxIterations;
        const double _minGradient;
        const double _convergenceThreshold;
        const double _dampingFactor;
        

};
}
#endif// VSLAM_RGBD_ODOMETRY

