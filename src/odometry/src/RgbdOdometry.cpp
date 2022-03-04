
#include "RgbdOdometry.h"

namespace pd::vision{

        RgbdOdometry::RgbdOdometry(double minGradient, int nLevels, int maxIterations, double convergenceThreshold, double dampingFactor)
        : _nLevels(nLevels)
        , _maxIterations(maxIterations)
        , _minGradient(minGradient)
        , _convergenceThreshold(convergenceThreshold)
        , _dampingFactor(dampingFactor)
        , _loss( std::make_unique<TukeyLoss> ())
        {
                Log::get("odometry",ODOMETRY_CFG_DIR"/log/odometry.conf");
        }

        PoseWithCovariance::UnPtr RgbdOdometry::align(FrameRgbd::ConstShPtr from, FrameRgb::ConstShPtr to) const
        {

                //TODO use covariance
                auto solver = std::make_unique<GaussNewton<LukasKanadeInverseCompositionalSE3>> ( 
                                _dampingFactor,
                                _convergenceThreshold,
                                _maxIterations);

                SE3d pose = to->pose().pose();
                for(int i = _nLevels; i > 0; i--)
                {
                        TIMED_SCOPE(timerI,"align at level ( " + std::to_string(i) + " )");

                        const auto s = 1.0/(double)i;
                        
                        auto templScaled = algorithm::resize(from->rgb(),s);
                        auto depthScaled = algorithm::resize(from->depth(),s);
                        auto imageScaled = algorithm::resize(to->rgb(),s);
                        
                        auto w = std::make_shared<WarpSE3>(pose.inverse(),depthScaled,
                        Camera::resize(from->camera(),s),Camera::resize(to->camera(),s),
                        from->pose().pose());

                        auto lk = std::make_shared<LukasKanadeInverseCompositionalSE3> (
                                templScaled,
                                imageScaled,
                                w,_loss,_minGradient);

                        solver->solve(lk);
                        
                        pose = w->SE3().inverse();
                    
                }
                return std::make_unique<PoseWithCovariance>( pose, solver->cov() );
        }
        PoseWithCovariance::UnPtr RgbdOdometry::align(std::vector<FrameRgbd::ConstShPtr>& from,  FrameRgb::ConstShPtr to) const
        {
                auto solver = (std::make_unique<GaussNewton<LukasKanadeInverseCompositionalStackedSE3>> ( 
                                _dampingFactor,
                                _convergenceThreshold,
                                _maxIterations));
              
                //TODO use covariance
                SE3d pose = to->pose().pose();
                for(int i = _nLevels; i > 0; i--)
                {
                        TIMED_SCOPE(timerI,"align at level ( " + std::to_string(i) + " )");
                       
                        const auto s = 1.0/(double)i;
                        auto imageScaled = algorithm::resize(to->rgb(),s);
                       
                        std::vector<Image> templs(from.size());
                        std::vector<DepthMap> depths(from.size());
                        std::vector<std::shared_ptr<WarpSE3>> warps(from.size());
                        for (size_t j = 0; j < from.size(); j++)
                        {
                                const auto s = 1.0/(double)i;
                        
                                templs[j] = algorithm::resize(from[j]->rgb(),s);
                                auto depthScaled = algorithm::resize(from[j]->depth(),s);
                                warps[j] = std::make_shared<WarpSE3>(pose.inverse(), depthScaled,
                                Camera::resize(from[j]->camera(),s), Camera::resize(to->camera(),s),
                                from[j]->pose().pose());

                                
                        }
                        auto lk = std::make_shared<LukasKanadeInverseCompositionalStackedSE3> (
                                        templs,
                                        imageScaled,
                                        warps,_loss,_minGradient);

                        solver->solve(lk);
                        
                        pose = warps[0]->SE3().inverse();
                        
                    
                }
                return std::make_unique<PoseWithCovariance>( pose, solver->cov() );
        }


}