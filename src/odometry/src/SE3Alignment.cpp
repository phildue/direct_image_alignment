
#include "SE3Alignment.h"

#define LOG_ODOM(level) CLOG(level,"odometry")
using namespace pd::vslam::solver;
using namespace pd::vslam::core;
namespace pd::vision{

        /*
        We expect the new pose not be too far away from a prediction.
        Namely we expect it to be normally distributed around the prediction ( mean ) with some uncertainty ( covariance ).
        */
        class MotionPrior : public Prior<6>{
        public:
                MotionPrior(const PoseWithCovariance& predictedPose, const PoseWithCovariance& referencePose)
                :Prior<6>()
                , _xPred((predictedPose.pose() * referencePose.pose().inverse()).log())
                ,_information(predictedPose.cov().inverse()){}
                
                void apply(NormalEquations<6>::ShPtr ne, const Eigen::VectorXd& x) const override{
                        const double normalizer = 1.0 / (255.0 * 255.0);//otherwise prior has no influence ?
                        ne->A.noalias() = ne->A * normalizer;
                        ne->b.noalias() = ne->b * normalizer;

                        ne->A.noalias() += _information; 
                        ne->b.noalias() += _information * (_xPred - x);

                        LOG_ODOM( DEBUG ) << "Prior: " << _xPred.transpose() << " \nInformation:\n " << _information;
                }
        private:
        Eigen::VectorXd _xPred;
        Eigen::MatrixXd _information;
        };

        SE3Alignment::SE3Alignment(double minGradient, Solver<6>::ShPtr solver, Loss::ShPtr loss, bool includePrior)
        : _minGradient2(minGradient*minGradient)
        , _loss( loss ) 
        , _solver ( solver )
        , _includePrior (includePrior)
        {
                Log::get("odometry",ODOMETRY_CFG_DIR"/log/odometry.conf");
        }


        PoseWithCovariance::UnPtr SE3Alignment::align(FrameRgbd::ConstShPtr from, FrameRgbd::ConstShPtr to) const
        {
                auto prior = _includePrior ? std::make_shared<MotionPrior>(to->pose(),from->pose()) : nullptr;
                SE3d pose = to->pose().pose();
                for(int level = from->nLevels()-1; level >= 0; level--)
                {
                        TIMED_SCOPE(timerI,"align at level ( " + std::to_string(level) + " )");
                        LOG_ODOM(INFO) << "Aligning from: \n"<< from->pose().pose().log().transpose() << " to " << pose.log().transpose()
                         << "\nat " << level << " image size: ["<< from->width(level) << "," << from->height(level) <<"].";

                        LOG_IMG("Image") << to->intensity(level);
                        LOG_IMG("Template") << from->intensity(level);
                        LOG_IMG("Depth") << from->depth(level);
                        
                        auto w = std::make_shared<WarpSE3>(
                                pose,from->pcl(level,false),from->width(level),
                                from->camera(level),to->camera(level),from->pose().pose());

                        std::vector<Eigen::Vector2i> interestPoints;
                        interestPoints.reserve(from->width(level)*from->height(level));
                        const MatXd gradientMagnitude = from->dIx(level).array().pow(2) + from->dIy(level).array().pow(2);
                        forEachPixel(gradientMagnitude,[&](int u, int v, double p)
                        {
                                if( p >= _minGradient2 && from->depth(level)(v,u) > 0.0)
                                {
                                        interestPoints.emplace_back(u,v);
                                }
                        });

                        auto lk = std::make_shared<LukasKanadeInverseCompositionalSE3> (
                                from->intensity(level),from->dIx(level), from->dIy(level),
                                to->intensity(level), w,interestPoints,
                                _loss, prior );

                        _solver->solve(lk);
                        
                        pose = w->poseCur();
                    
                }
                return std::make_unique<PoseWithCovariance>( pose, MatXd::Identity(6,6) );
        }
        PoseWithCovariance::UnPtr SE3Alignment::align(const std::vector<FrameRgbd::ConstShPtr>& from,  FrameRgbd::ConstShPtr to) const
        {
                SE3d pose;
                for(int level = from[0]->nLevels()-1; level >= 0; level--)
                {
                        TIMED_SCOPE(timerI,"align at level ( " + std::to_string(level) + " )");
                       
                        std::vector<std::shared_ptr<LukasKanadeInverseCompositionalSE3>> frames;
                        for (const auto& f : from)
                        {
                                auto prior = _includePrior ? std::make_shared<MotionPrior>(to->pose(),f->pose()) : nullptr;
                                
                                auto w = std::make_shared<WarpSE3>(pose,f->depth(level),
                                     f->camera(level),to->camera(level),f->pose().pose());

                                std::vector<Eigen::Vector2i> interestPoints;
                                interestPoints.reserve(f->width(level)*f->height(level));
                                const MatXd gradientMagnitude = f->dIx(level).array().pow(2) + f->dIy(level).array().pow(2);
                                forEachPixel(gradientMagnitude,[&](int u, int v, double p)
                                {
                                        if( p >= _minGradient2 && f->depth(level)(v,u) > 0.0)
                                        {
                                                interestPoints.emplace_back(u,v);
                                        }
                                });

                                vslam::solver::Problem<6>::ShPtr lk = std::make_shared<LukasKanadeInverseCompositionalSE3> (
                                f->intensity(level),f->dIx(level), f->dIy(level),
                                to->intensity(level), w,interestPoints, _loss, prior );

                        }
                        vslam::solver::Problem<6>::ShPtr lk = std::make_shared<LukasKanadeInverseCompositionalStackedSE3> (frames);

                        _solver->solve(lk);
                        
                        pose = frames[0]->warp()->poseCur();
                        
                    
                }
                return std::make_unique<PoseWithCovariance>( pose, MatXd::Identity(6,6) );
        }


}