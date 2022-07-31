#include "BundleAdjustment.h"
#include <sophus/ceres_manifold.hpp>
#include "utils/utils.h"
#define LOG_BA(level) CLOG(level,"bundle_adjustment")

namespace pd::vslam::mapping{

        class ReprojectionErrorManifold {
                public:

                ReprojectionErrorManifold(const Eigen::Vector2d& observation, const Eigen::Matrix3d& K) :
                _obs{observation}, _K{K} {}

                template <typename T>
                bool operator()(const T* const poseData, const T* const pointData, T* residuals) const
                {
                        Eigen::Map<Sophus::SE3<T> const> const pose(poseData);
                        Eigen::Map<Eigen::Matrix<T,3,1> const> pWcs(pointData);

                        auto pCcs = pose * pWcs;

                        if (pCcs[2] > 0.1 )
                        {
                                auto pIcs = _K * pCcs;

                                residuals[0] = pIcs.x()/pIcs.z() - T(_obs.x());
                                residuals[1] = pIcs.y()/pIcs.z() - T(_obs.y());
                        }else{
                                residuals[0] = T(0.0);
                                residuals[1] = T(0.0);
                        }
                        return true;
                }
                static ceres::CostFunction *Create(const Eigen::Vector2d& observation,
                                                const Eigen::Matrix3d& K
                                                ) {
                        return (new ceres::AutoDiffCostFunction<ReprojectionErrorManifold, 2, Sophus::SE3d::num_parameters, 3>(
                        new ReprojectionErrorManifold(observation,K)));
                }

                private:
                const Eigen::Vector2d _obs;
                const Eigen::Matrix3d _K;
        };
        BundleAdjustment::BundleAdjustment(const std::vector<FrameRgb::ConstShPtr>& frames, const std::vector<Point3D::ConstShPtr>& points)
        :_frames(frames){
                Log::get("bundle_adjustment",ODOMETRY_CFG_DIR"/log/bundle_adjustment.conf");
                
                for (const auto& f : frames) 
                {
                        _poses[f->id()] = f->pose().pose();
                        _problem.AddParameterBlock(_poses[f->id()].data(),SE3d::num_parameters,new Sophus::Manifold<Sophus::SE3>());
                }
                for (const auto& p : points) 
                {
                        _points[p->id()] = p->position();
                        for (const auto ft : p->features())
                        {
                                _problem.AddResidualBlock(ReprojectionErrorManifold::Create(
                                                ft->position(),
                                                ft->frame()->camera()->K()),
                                        nullptr /* squared loss */,
                                        _poses[ft->frame()->id()].data(),
                                        _points[p->id()].data());
                        }
                }

        }
        void BundleAdjustment::optimize()
        {
                ceres::Solver::Options options;
                options.linear_solver_type = ceres::DENSE_SCHUR;
                options.minimizer_progress_to_stdout = true;
                options.max_num_iterations = 200;
                //double errorPrev = computeReprojectionError();
                ceres::Solver::Summary summary;
                ceres::Solve(options, &_problem, &summary);

                LOG_BA(DEBUG) << summary.FullReport();
                //double errorAfter = computeReprojectionError();
                //std::cout << "Before: " << errorPrev << " -->  " << errorAfter << std::endl;
        }

        void BundleAdjustment::update(const std::vector<FrameRgb::ShPtr>& frames) const
        {
                for(const auto& f : frames)
                {
                        f->set(PoseWithCovariance(_poses.find(f->id())->second,Matd<6,6>::Identity()));
                }
        }
        void BundleAdjustment::update(const std::vector<Point3D::ShPtr>& points) const
        {
                for(const auto& p : points)
                {
                        p->position() = _points.find(p->id())->second;
                }
        }

        double BundleAdjustment::computeError() const
        {
                double error = 0.0;
                for (const auto& id_pose : _poses) 
                {
                        const auto& f = _frames[id_pose.first];
                        for (const auto& ft : f->features()) 
                        {
                                if(ft->point())
                                {
                                        auto pCcs = id_pose.second * _points.find(ft->point()->id())->second;
                                        if (pCcs.z() > 0.1)
                                        {
                                                Eigen::Vector2d pIcs = f->camera2image(pCcs);
                                                        auto e = ft->position() - pIcs;
                                                        error += e.norm();
                                        }
                                }
                               
                        }
                }
                return error;
        }

}