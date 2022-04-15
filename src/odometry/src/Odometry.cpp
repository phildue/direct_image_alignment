
#include "Odometry.h"
#define LOG_ODOM(level) CLOG(level,"odometry")
namespace pd::vision{
         //:_aligner(std::make_shared<SE3Alignment>(minGradient,solver,loss,scaler))

        OdometryRgbd::OdometryRgbd(double minGradient,
         vslam::solver::Solver<6>::ShPtr solver,
         vslam::solver::Loss::ShPtr loss,
         Map::ConstShPtr map)
         :_aligner(std::make_shared<SE3Alignment>(minGradient,solver,loss,true))
         ,_map(map)
         ,_includeKeyFrame(true)
         {
                Log::get("odometry",ODOMETRY_CFG_DIR"/log/odometry.conf");

         }
        void OdometryRgbd::update(FrameRgbd::ConstShPtr frame)
        {
                if(_map->lastFrame())
                {
                        try{
                                if(_includeKeyFrame && _map->lastKf() != nullptr)
                                {
                                _pose = _aligner->align({_map->lastKf(),_map->lastFrame()},frame);
                                        
                                }else{
                                        _pose = _aligner->align(_map->lastFrame(),frame);
                                }
                                auto dT = frame->t() - _map->lastFrame()->t();
                                _speed = std::make_shared<PoseWithCovariance>(SE3d::exp(algorithm::computeRelativeTransform(_map->lastFrame()->pose().pose(),_pose->pose()).log()/((double)dT/1e9)),_pose->cov());
   
                        }catch(const std::runtime_error& e){
                                LOG_ODOM( ERROR ) << e.what();
                                _pose = std::make_shared<PoseWithCovariance>(frame->pose());
                                _speed = std::make_shared<PoseWithCovariance>();
                                
                        }
                }else{
                        _pose = std::make_shared<PoseWithCovariance>(frame->pose());
                        _speed = std::make_shared<PoseWithCovariance>();
                        LOG_ODOM( DEBUG ) << "Processing first frame";
                }
        }

        OdometryIcp::OdometryIcp(int level, int maxIterations, Map::ConstShPtr map)
         :_aligner(std::make_shared<IterativeClosestPoint>(level, maxIterations))
         ,_map(map)
         {}
        void OdometryIcp::update(FrameRgbd::ConstShPtr frame)
        {
                if(_map->lastFrame())
                {       
                        LOG_ODOM( DEBUG ) << "Processing frame";
                        _pose = _aligner->align(_map->lastFrame(),frame);
                        auto dT = frame->t() - _map->lastFrame()->t();
                        _speed = std::make_shared<PoseWithCovariance>(
                                SE3d::exp(
                                algorithm::computeRelativeTransform(_map->lastFrame()->pose().pose(),_pose->pose()).log()/((double)dT/1e9)),
                                _pose->cov());
                       
                }else{
                        _pose = std::make_shared<PoseWithCovariance>(frame->pose());
                        _speed = std::make_shared<PoseWithCovariance>();
                        LOG_ODOM( DEBUG ) << "Processing first frame";
                }
        }       

        
}