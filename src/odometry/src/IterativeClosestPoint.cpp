
#include <utils/utils.h>
#include "IterativeClosestPoint.h"
#define LOG_ODOM(level) CLOG(level,"odometry")
namespace pd::vision{


        PoseWithCovariance::UnPtr IterativeClosestPoint::align(RgbdPyramid::ConstShPtr from, RgbdPyramid::ConstShPtr to) const
        {
                /*TODO move to Frame class*/
                pcl::PointCloud<pcl::PointXYZ>::Ptr cloudFrom (new pcl::PointCloud<pcl::PointXYZ>);

                for(int v = 0; v < from->depth().rows(); v++)
                {
                        for(int u = 0; u < from->depth().cols(); u++)
                        {
                                if (std::isfinite(from->depth(0)(v,u)) && from->depth(0)(v,u) > 0 )
                                {
                                        auto p = from->pose().pose() * from->camera(0)->image2camera({u,v},from->depth(0)(v,u));
                                        cloudFrom->emplace_back(p.x(),p.y(),p.z());
                                }
                        }
                }
                pcl::PointCloud<pcl::PointXYZ>::Ptr cloudTo (new pcl::PointCloud<pcl::PointXYZ>);

                for(int v = 0; v < to->depth().rows(); v++)
                {
                        for(int u = 0; u < to->depth().cols(); u++)
                        {
                                if (std::isfinite(to->depth(0)(v,u)) && to->depth(0)(v,u) > 0 )
                                {
                                        auto p = to->pose().pose() * to->camera(0)->image2camera({u,v},to->depth(0)(v,u));
                                        cloudTo->emplace_back(p.x(),p.y(),p.z());
                                }
                        }
                }
                pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
                icp.setMaximumIterations (_maxIterations);
                icp.setInputSource (cloudFrom);
                icp.setInputTarget (cloudTo);
                icp.align (*cloudTo);
                PoseWithCovariance::UnPtr pose_;
                if (icp.hasConverged ())
                {
                        LOG_ODOM(INFO) << "ICP has converged, score is" << icp.getFitnessScore ();
                        pose_ = std::make_unique<PoseWithCovariance>(SE3d(icp.getFinalTransformation ().cast<double>()) * from->pose().pose(),MatXd::Identity(6,6));
                }
                else
                {
                        LOG_ODOM(ERROR) << "ICP has not converged";
                        pose_ = std::make_unique<PoseWithCovariance>(from->pose());
                }
                return pose_;
        }


}