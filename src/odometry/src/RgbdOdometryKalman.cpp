
#include "RgbdOdometryKalman.h"
#include "lukas_kanade/lukas_kanade.h"
#include "solver/solver.h"
#define LOG_ODOM(level) CLOG(level, "odometry")
namespace pd::vision{


        RgbdOdometryKalman::RgbdOdometryKalman(Camera::ShPtr camera, double minGradient, int nLevels, int maxIterations, double convergenceThreshold, double dampingFactor)
        :_filter(std::make_unique<KalmanFilterSE3>( Matd<6,6>::Identity() * 1.1, Matd<6,1>::Zero()))
        ,_aligner(std::make_unique<RgbdOdometry>( camera, minGradient, nLevels, maxIterations, convergenceThreshold, dampingFactor ))
        {
                Log::get("odometry");

        }
       
        SE3d RgbdOdometryKalman::estimate(const Image& fromRgb,const DepthMap& fromDepth, const Image& toRgb, std::uint64_t t) const
        {
                random::Gaussian<6> dPose{ Matd<6,6>::Identity(), Matd<6,1>::Zero() };
                auto pred = _filter->predict(t);
                dPose.mean = {pred.state(0),pred.state(1),pred.state(2),pred.state(3),pred.state(4),pred.state(5)};
                dPose.cov = pred.cov.block(0,0,6,6);

                LOG_ODOM(DEBUG) << "Predicted:" << dPose.mean.transpose();

                dPose.mean = _aligner->estimate(fromRgb, fromDepth, toRgb, t, SE3d::exp(dPose.mean) ).log();
                
                LOG_ODOM(DEBUG) << "Estimated:" << dPose.mean.transpose();
                
                _filter->update(t, dPose.mean, dPose.cov);

                return SE3d::exp(dPose.mean);
        }
}