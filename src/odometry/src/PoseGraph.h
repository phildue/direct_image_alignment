#ifndef VSLAM_ODOMETRY_POSE_GRAPH_H__
#define VSLAM_ODOMETRY_POSE_GRAPH_H__
#include <map>
#include <core/core.h>

namespace pd::vslam::odometry
{
        class PoseGraph{
                public:
                PoseGraph(const std::map<vision::Timestamp,vision::PoseWithCovariance::ConstShPtr>& poses);
                PoseGraph(const std::map<vision::Timestamp,vision::SE3d>& poses);
                vision::PoseWithCovariance::ConstShPtr poseAt(vision::Timestamp t, bool interpolate = true) const;
                vision::PoseWithCovariance::ConstShPtr poseBetween(vision::Timestamp t0,vision::Timestamp t1, bool interpolate = true) const;
                void append(vision::Timestamp t, vision::PoseWithCovariance::ConstShPtr pose);
                const std::map<vision::Timestamp, vision::PoseWithCovariance::ConstShPtr>& poses() const {return _poses;}
                private:
                
                vision::PoseWithCovariance::ConstShPtr interpolateAt(vision::Timestamp t) const;

                std::map<vision::Timestamp, vision::PoseWithCovariance::ConstShPtr> _poses;

        };
} // namespace pd::vslam::odometry

#endif