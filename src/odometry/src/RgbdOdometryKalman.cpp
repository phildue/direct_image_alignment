
#include "RgbdOdometryKalman.h"
#include "lukas_kanade/lukas_kanade.h"
#include "solver/solver.h"
namespace pd::vision{

        class KalmanFilterMotion : public KalmanFilter
        {
        ///state = [vx,vy,vz, avx,avy,avz]
        ///measurement = [vx,vy,vz, avx,avy,avz]
        public:
        KalmanFilterMotion()
        : KalmanFilter(MatD::Zero(6,1),MatD::Zero(6,6),MatD::Identity(6,6),VecD::Zero(6),0U)
        {
                _P.noalias() = MatD::Identity(6,6) * 0.1;
        }
        MatD A(std::uint64_t dT) const override
        {
                return MatD::Identity(6,6);
        }
        };

        RgbdOdometryKalman::RgbdOdometryKalman(Camera::ShPtr camera, double minGradient, int nLevels, int maxIterations, double convergenceThreshold, double dampingFactor)
        : RgbdOdometry(camera,minGradient,nLevels,maxIterations,convergenceThreshold,dampingFactor)
        , _kalman(std::make_shared<KalmanFilterMotion>())
        {}

        SE3d RgbdOdometryKalman::estimate(const Image& fromRgb,const DepthMap& fromDepth, const Image& toRgb, std::uint64_t t)
        {
                Sophus::SE3d dPose;
                auto l = std::make_shared<HuberLoss>(10);

                Eigen::VectorXd xPred = _kalman->predict(t);
                dPose = Sophus::SE3d::exp({xPred(0),xPred(1),xPred(2),xPred(3),xPred(4),xPred(5)});
                const MatD& cov = _kalman->cov();
                //TODO include prediction as a prior

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
                _kalman->update(t,dPose.log(),solver->covScaled());
                return dPose;
        }
}