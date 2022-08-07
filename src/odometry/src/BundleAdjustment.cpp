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
        BundleAdjustment::BundleAdjustment()
        {
                Log::get("mapping",ODOMETRY_CFG_DIR"/log/mapping.conf");

        }
        void BundleAdjustment::optimize(const std::vector<FrameRgbd::ShPtr>& frames, const std::vector<Point3D::ShPtr>& points)
        {
                insertFrames(frames.begin(),frames.end());
                insertPoints(points.begin(),points.end());
                optimize();//new thread ?
                getPoses(frames.begin(),frames.end());
                getPositions(points.begin(),points.end());
        }

        void BundleAdjustment::insertFrame(std::uint64_t frameId, const SE3d& pose, const Mat3d& K)
        {
                _poses[frameId] = pose;
                _Ks[frameId] = K;
                _problem.AddParameterBlock(_poses[frameId].data(),SE3d::num_parameters,new Sophus::Manifold<Sophus::SE3>());
  
        }
        void BundleAdjustment::insertPoint(std::uint64_t pointId, const Vec3d& position)
        {
                _points[pointId] = position;

        }
        void BundleAdjustment::insertObservation(std::uint64_t pointId, std::uint64_t frameId, const Vec2d& observation)
        {
                auto itF = _Ks.find(frameId);
                if (itF == _Ks.end()){ throw pd::Exception("No corresponding frame found.");}
                auto itP = _points.find(pointId);
                if (itP == _points.end()){ throw pd::Exception("No corresponding point found.");}

                auto& K = itF->second;
                auto& pose = _poses.find(frameId)->second;
                auto& point = itP->second;
                 _problem.AddResidualBlock(ReprojectionErrorManifold::Create(
                                        observation,
                                        K),
                                nullptr /* squared loss */,
                                pose.data(),
                                point.data());
        }
        
        void BundleAdjustment::insertFrame(FrameRgb::ConstShPtr f)
        {
                _frames.push_back(FrameRgb::ConstShPtr(f));
                insertFrame(f->id(),f->pose().pose(),f->camera()->K());
  
        }
        void BundleAdjustment::insertPoint(Point3D::ConstShPtr p)
        {
                insertPoint(p->id(),p->position());
                std::for_each(p->features().begin(),p->features().end(),
                [&](auto ft){ insertObservation(p->id(),ft->frame()->id(),ft->position());});
        }
        
        PoseWithCovariance::UnPtr BundleAdjustment::getPose(std::uint64_t frameId) const
        {
                auto it = _poses.find(frameId);
                if(it == _poses.end()){ throw pd::Exception("Frame was not optimized;");}
                return std::make_unique<PoseWithCovariance>(it->second,Matd<6,6>::Identity());
        }
        Vec3d BundleAdjustment::getPosition(std::uint64_t pointId) const
        {
                auto it = _points.find(pointId);
                if(it == _points.end()){ throw pd::Exception("Point was not optimized;");}
                return it->second;
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