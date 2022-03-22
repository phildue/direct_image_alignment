
#include "SE3Alignment.h"
#define LOG_ODOM(level) CLOG(level,"odometry")
namespace pd::vision{

        SE3Alignment::SE3Alignment(double minGradient, vslam::solver::Solver<6>::ShPtr solver, vslam::solver::Loss::ShPtr loss, vslam::solver::Scaler::ShPtr scaler)
        : _minGradient(minGradient)
        , _loss( loss ) 
        , _scaler ( scaler )
        , _solver ( solver )
        {
                Log::get("odometry",ODOMETRY_CFG_DIR"/log/odometry.conf");
        }


        PoseWithCovariance::UnPtr SE3Alignment::align(RgbdPyramid::ConstShPtr from, RgbdPyramid::ConstShPtr to) const
        {

                //TODO use covariance
                SE3d pose = to->pose().pose();
                for(int level = from->nLevels()-1; level >= 0; level--)
                {
                        TIMED_SCOPE(timerI,"align at level ( " + std::to_string(level) + " )");

                        LOG_IMG("Image") << to->intensity(level);
                        LOG_IMG("Template") << from->intensity(level);
                        LOG_IMG("Depth") << from->depth(level);
                        
                        auto w = std::make_shared<WarpSE3>(pose.inverse(),from->depth(level),
                        from->camera(level),to->camera(level),from->pose().pose());

                        vslam::solver::Problem<6>::ShPtr lk = std::make_shared<LukasKanadeInverseCompositionalSE3> (
                                from->intensity(level),
                                from->dIx(level), from->dIy(level),
                                to->intensity(level),
                                w,
                                _loss,
                                _minGradient,
                                _scaler);

                        _solver->solve(lk);
                        
                        pose = w->SE3().inverse();
                        LOG_ODOM(INFO) << "Aligned level: "<< level << " image size: ["<< from->intensity(level).cols() << "," << from->intensity(level).rows() <<"].";
                    
                }
                return std::make_unique<PoseWithCovariance>( pose, _solver->cov() );
        }
        PoseWithCovariance::UnPtr SE3Alignment::align(const std::vector<RgbdPyramid::ConstShPtr>& from,  RgbdPyramid::ConstShPtr to) const
        {
                //TODO use covariance
                SE3d pose = to->pose().pose();
                for(int level = from[0]->nLevels()-1; level >= 0; level--)
                {
                        TIMED_SCOPE(timerI,"align at level ( " + std::to_string(level) + " )");
                       
                        std::vector<std::shared_ptr<LukasKanadeInverseCompositionalSE3>> frames;
                        for (const auto& f : from)
                        {
                        
                                auto w = std::make_shared<WarpSE3>(pose.inverse(),f->depth(level),
                                f->camera(level),to->camera(level),f->pose().pose());

                                frames.emplace_back(std::make_shared<LukasKanadeInverseCompositionalSE3>(
                                        f->intensity(level),f->dIx(level), f->dIy(level),to->intensity(level),w,
                                        _loss,
                                        _minGradient,
                                        _scaler));

                                
                        }
                        vslam::solver::Problem<6>::ShPtr lk = std::make_shared<LukasKanadeInverseCompositionalStackedSE3> (frames);

                        _solver->solve(lk);
                        
                        pose = frames[0]->warp()->SE3().inverse();
                        
                    
                }
                return std::make_unique<PoseWithCovariance>( pose, _solver->cov() );
        }


}