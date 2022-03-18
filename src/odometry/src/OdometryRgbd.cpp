
#include "OdometryRgbd.h"
#define LOG_ODOM(level) CLOG(level,"odometry")
namespace pd::vision{

        OdometryRgbd::OdometryRgbd(double minGradient,  const std::vector<double>& levels, int maxIterations, double convergenceThreshold, vslam::solver::Loss::ShPtr loss, vslam::solver::Scaler::ShPtr scaler, Map::ConstShPtr map, double dampingFactor)
        : _maxIterations(maxIterations)
        , _minGradient(minGradient)
        , _convergenceThreshold(convergenceThreshold)
        , _dampingFactor(dampingFactor)
        , _loss( loss ) 
        , _scaler ( scaler )
        , _levels(levels)
        , _lastFrame(nullptr)
        , _speed(std::make_shared<PoseWithCovariance>(SE3d(),MatXd::Identity(6,6)))
        , _map(map)
        {
                Log::get("odometry",ODOMETRY_CFG_DIR"/log/odometry.conf");
        }
        void OdometryRgbd::update(FrameRgbd::ConstShPtr frame)
        {
                if(_lastFrame)
                {
                        LOG_IMG("Image") << frame->rgb();
                        LOG_IMG("Template") << _lastFrame->rgb();
                        LOG_IMG("Depth") << _lastFrame->depth();
                        PoseWithCovariance::ConstShPtr pose_;
                        if(_map->lastKf() != nullptr)
                        {
                                pose_ = align({_map->lastKf(),_lastFrame},frame);

                        }else{
                                pose_ = align(_lastFrame,frame);
                        }
                        auto dT = frame->t() - _lastFrame->t();
                        _speed = std::make_shared<PoseWithCovariance>(SE3d::exp(algorithm::computeRelativeTransform(_lastFrame->pose().pose(),pose_->pose()).log()/((double)dT/1e9)),pose_->cov());
                        _lastFrame = std::make_shared<const pd::vision::FrameRgbd>(frame->rgb(),frame->depth(),frame->camera(),frame->t(), *pose_);
                }else{
                        _lastFrame = frame;
                }
        }


        PoseWithCovariance::UnPtr OdometryRgbd::align(FrameRgbd::ConstShPtr from, FrameRgb::ConstShPtr to) const
        {

                //TODO use covariance
                auto solver = std::make_unique<vslam::solver::GaussNewton<LukasKanadeInverseCompositionalSE3>> ( 
                                _dampingFactor,
                                _convergenceThreshold,
                                _maxIterations);

                SE3d pose = to->pose().pose();
                for(const auto& level : _levels)
                {
                        TIMED_SCOPE(timerI,"align at level ( " + std::to_string(level) + " )");

                        const double s = level;
                        //TODO move this to an "Image pyramid class"
                        auto templScaled = algorithm::resize(from->rgb(),s);
                        auto depthScaled = algorithm::resize(from->depth(),s);
                        auto imageScaled = algorithm::resize(to->rgb(),s);
                        auto dTx = algorithm::conv2d<int>(templScaled.cast<int>(),Kernel2d<int>::dX());
                        auto dTy = algorithm::conv2d<int>(templScaled.cast<int>(),Kernel2d<int>::dY());
                        
                        auto w = std::make_shared<WarpSE3>(pose.inverse(),depthScaled,
                        Camera::resize(from->camera(),s),Camera::resize(to->camera(),s),
                        from->pose().pose());

                        auto lk = std::make_shared<LukasKanadeInverseCompositionalSE3> (
                                templScaled,
                                dTx, dTy,
                                imageScaled,
                                w,
                                _loss,
                                _minGradient,
                                _scaler);

                        solver->solve(lk);
                        
                        pose = w->SE3().inverse();
                        LOG_ODOM(INFO) << "Aligned level: "<< s << " image size: ["<< templScaled.cols() << "," << templScaled.rows() <<"]";
                    
                }
                return std::make_unique<PoseWithCovariance>( pose, solver->cov() );
        }
        PoseWithCovariance::UnPtr OdometryRgbd::align(const std::vector<FrameRgbd::ConstShPtr>& from,  FrameRgb::ConstShPtr to) const
        {
                auto solver = (std::make_unique<vslam::solver::GaussNewton<LukasKanadeInverseCompositionalStackedSE3>> ( 
                                _dampingFactor,
                                _convergenceThreshold,
                                _maxIterations));
              
                //TODO use covariance
                SE3d pose = to->pose().pose();
                for(const auto& level : _levels)
                {
                        TIMED_SCOPE(timerI,"align at level ( " + std::to_string(level) + " )");
                       
                        const auto s = level;
                        auto imageScaled = algorithm::resize(to->rgb(),s);
                       
                        std::vector<Image> templs(from.size());
                        std::vector<MatXi> dTxs(from.size());
                        std::vector<MatXi> dTys(from.size());
                        std::vector<DepthMap> depths(from.size());
                        std::vector<std::shared_ptr<WarpSE3>> warps(from.size());
                        for (size_t j = 0; j < from.size(); j++)
                        {
                        
                                templs[j] = algorithm::resize(from[j]->rgb(),s);
                                dTxs[j] = algorithm::conv2d<int>(templs[j].cast<int>(),Kernel2d<int>::dX());
                                dTys[j] = algorithm::conv2d<int>(templs[j].cast<int>(),Kernel2d<int>::dY());
                    
                                auto depthScaled = algorithm::resize(from[j]->depth(),s);
                                warps[j] = std::make_shared<WarpSE3>(pose.inverse(), depthScaled,
                                Camera::resize(from[j]->camera(),s), Camera::resize(to->camera(),s),
                                from[j]->pose().pose());

                                
                        }
                        auto lk = std::make_shared<LukasKanadeInverseCompositionalStackedSE3> (
                                        templs,
                                        dTxs,dTys,
                                        imageScaled,
                                        warps,_loss,_minGradient,_scaler);

                        solver->solve(lk);
                        
                        pose = warps[0]->SE3().inverse();
                        
                    
                }
                return std::make_unique<PoseWithCovariance>( pose, solver->cov() );
        }


}