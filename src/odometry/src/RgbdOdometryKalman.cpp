
#include "RgbdOdometryKalman.h"
#include "lukas_kanade/lukas_kanade.h"
#include "solver/solver.h"
#define LOG_ODOM(level) CLOG(level, "odometry")
namespace pd::vision{


        RgbdOdometryKalman::RgbdOdometryKalman(double minGradient, int nLevels, int maxIterations, double convergenceThreshold, double dampingFactor)
        :_filter(std::make_unique<KalmanFilterSE3>( Matd<6,6>::Identity() * 1.1, Matd<6,1>::Zero()))
        ,_aligner(std::make_unique<RgbdOdometry>(minGradient, nLevels, maxIterations, convergenceThreshold, dampingFactor ))
        {
                Log::get("odometry");

        }
       
        PoseWithCovariance::UnPtr RgbdOdometryKalman::align(FrameRgbd::ConstShPtr from, FrameRgb::ConstShPtr to) const
        {
                auto pred = _filter->predict(to->t());
                auto pose = std::make_unique<PoseWithCovariance>(pred.state.head(6),pred.cov.block(0,0,6,6));

                LOG_ODOM(DEBUG) << "Predicted:" << pose->mean().transpose();
                auto toPred = std::make_shared<FrameRgb>(to->rgb(),to->camera(), to->t(), *pose);
                pose = _aligner->align(from, toPred );
                
                LOG_ODOM(DEBUG) << "Estimated:" << pose->mean().transpose();
                
                _filter->update(to->t(), pose->mean(), pose->cov() );

                return pose;
        }
}