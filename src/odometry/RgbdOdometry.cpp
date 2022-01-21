
#include "RgbdOdometry.h"
#include "lukas_kanade/LukasKanadeInverseCompositional.h"
#include "solver/GaussNewton.h"
namespace pd::vision{

        RgbdOdometry::RgbdOdometry(Camera::ShPtr camera, double minGradient, int nLevels, int maxIterations, double convergenceThreshold, double dampingFactor)
        : _camera(camera)
        , _nLevels(nLevels)
        , _maxIterations(maxIterations)
        , _minGradient(minGradient)
        , _convergenceThreshold(convergenceThreshold)
        , _dampingFactor(dampingFactor)
        {}

        SE3d RgbdOdometry::estimate(const Image& fromRgb,const DepthMap& fromDepth, const Image& toRgb)
        {
                Sophus::SE3d dPose;
                auto l = std::make_shared<HuberLoss>(10);
                auto solver = std::make_shared<GaussNewton<LukasKanadeInverseCompositionalSE3>> ( 
                                _dampingFactor,
                                _convergenceThreshold,
                                _maxIterations);
                
                for(int i = _nLevels; i > 0; i--)
                {
                        TIMED_SCOPE(timerI,"align at level ( " + std::to_string(i) + " )");

                        const auto s = 1.0/(double)i;
                        
                        auto templScaled = algorithm::resize(fromRgb,s);
                        auto depthScaled = algorithm::resize(fromDepth,s);
                        auto imageScaled = algorithm::resize(toRgb,s);
                        auto w = std::make_shared<WarpSE3>(dPose.log(),depthScaled,Camera::resize(_camera,s));

                        auto lk = std::make_shared<LukasKanadeInverseCompositionalSE3> (
                                templScaled,
                                imageScaled,
                                w,l,_minGradient);

                        solver->solve(lk);
                        
                        dPose = w->pose().inverse();
                    
                }
                return dPose;
        }
}