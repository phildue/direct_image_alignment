#ifndef VSLAM_RGBD_ODOMETRY
#define VSLAM_RGBD_ODOMETRY
#include "core/types.h"
#include "core/Camera.h"

namespace pd::vision{
class RgbdOdometry{
        public:
        RgbdOdometry(Camera::ShPtr camera, double minGradient = 50, int nLevels = 4)
        : _camera(camera)
        , _minGradient(minGradient)
        , _nLevels(nLevels)
        {}
        SE3d estimate(const Image& fromRgb,const DepthMap& fromDepth, const Image& toRgb);
        private:
        const int _nLevels;
        const double _minGradient;
        const Camera::ShPtr _camera;
        

};
}
#endif// VSLAM_RGBD_ODOMETRY

