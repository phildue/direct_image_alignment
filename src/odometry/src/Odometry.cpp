
#include "Odometry.h"
#define LOG_ODOM(level) CLOG(level,"odometry")
namespace pd::vision{
        OdometryRgbd::OdometryRgbd(double minGradient,
         vslam::solver::Solver<6>::ShPtr solver,
         vslam::solver::Loss::ShPtr loss,
         vslam::solver::Scaler::ShPtr scaler,
         Map::ConstShPtr map)
         :_aligner(std::make_shared<SE3Alignment>(minGradient,solver,loss,scaler))
         ,_map(map)
         ,_includeKeyFrame(false)
         {}
        void OdometryRgbd::update(RgbdPyramid::ConstShPtr frame)
        {
                if(_lastFrame)
                {
                        PoseWithCovariance::ConstShPtr pose_;
                        if(_includeKeyFrame && _map->lastKf() != nullptr)
                        {
                                pose_ = _aligner->align({_map->lastKf(),_lastFrame},frame);
                                
                        }else{
                                pose_ = _aligner->align(_lastFrame,frame);
                        }
                        auto dT = frame->t() - _lastFrame->t();
                        _speed = std::make_shared<PoseWithCovariance>(SE3d::exp(algorithm::computeRelativeTransform(_lastFrame->pose().pose(),pose_->pose()).log()/((double)dT/1e9)),pose_->cov());
                        _lastFrame = std::make_shared<const pd::vision::RgbdPyramid>(frame->intensity(),frame->depth(),frame->camera(),frame->scales(), frame->t(), *pose_);
                }else{
                        _lastFrame = frame;
                }
        }
        
}