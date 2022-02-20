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

        RgbdOdometry(Camera::ShPtr camera, double minGradient = 50, int nLevels = 4, int maxIterations = 20, double convergenceThreshold = 1e-4, double dampingFactor = 1.0);

        SE3d estimate(const Image& fromRgb,const DepthMap& fromDepth, const Image& toRgb, std::uint64_t t, const SE3d& p0 = {}) const;

        protected:
        const Camera::ShPtr _camera;
        const int _nLevels;
        const int _maxIterations;
        const double _minGradient;
        const double _convergenceThreshold;
        const double _dampingFactor;
        const GaussNewton<LukasKanadeInverseCompositionalSE3>::UnPtr _solver;
        const Loss::ShPtr _loss;
        

};
}
#endif// VSLAM_RGBD_ODOMETRY

