#ifndef VSLAM_RGBD_ODOMETRY
#define VSLAM_RGBD_ODOMETRY

#include "core/core.h"
#include "lukas_kanade/lukas_kanade.h"
#include "solver/solver.h"
#include "Odometry.h"
#include "Map.h"
namespace pd::vision{
class OdometryRgbd : public Odometry{
        public:
        typedef std::shared_ptr<OdometryRgbd> ShPtr;
        typedef std::unique_ptr<OdometryRgbd> UnPtr;
        typedef std::shared_ptr<const OdometryRgbd> ConstShPtr;
        typedef std::unique_ptr<const OdometryRgbd> ConstUnPtr;

        OdometryRgbd(double minGradient,
         const std::vector<double>& levels,
         int maxIterations, double convergenceThreshold,
         vslam::solver::Loss::ShPtr loss,
         vslam::solver::Scaler::ShPtr scaler,
         Map::ConstShPtr map,
         double dampingFactor = 1.0
         );

        void update(FrameRgbd::ConstShPtr frame) override;
        
        PoseWithCovariance::ConstShPtr pose() const override { return std::make_shared<const PoseWithCovariance>(_lastFrame->pose());}
        PoseWithCovariance::ConstShPtr speed() const override { return _speed;}

   
        protected:
        PoseWithCovariance::UnPtr align(FrameRgbd::ConstShPtr from, FrameRgb::ConstShPtr to) const;
        PoseWithCovariance::UnPtr align(const std::vector<FrameRgbd::ConstShPtr>& from,  FrameRgb::ConstShPtr to) const;

        const int _maxIterations;
        const double _minGradient;
        const double _convergenceThreshold;
        const double _dampingFactor;
        const vslam::solver::Loss::ShPtr _loss;
        const vslam::solver::Scaler::ShPtr _scaler;
        const std::vector< double > _levels;
        FrameRgbd::ConstShPtr _lastFrame;
        PoseWithCovariance::ConstShPtr _speed;
        Map::ConstShPtr _map;

};
}
#endif// VSLAM_RGBD_ODOMETRY

