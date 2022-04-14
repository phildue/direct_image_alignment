#ifndef VSLAM_ODOMETRY_POSE_GRAPH_H__
#define VSLAM_ODOMETRY_POSE_GRAPH_H__
#include <map>
#include <core/core.h>

namespace pd::vslam::odometry
{
        class Trajectory{
                public:
                typedef std::shared_ptr<Trajectory> ShPtr;
                typedef std::unique_ptr<Trajectory> UnPtr;
                typedef std::shared_ptr<const Trajectory> ConstShPtr;
                typedef std::unique_ptr<const Trajectory> ConstUnPtr;

                Trajectory(const std::map<vision::Timestamp,vision::PoseWithCovariance::ConstShPtr>& poses);
                Trajectory(const std::map<vision::Timestamp,vision::SE3d>& poses);
                vision::PoseWithCovariance::ConstShPtr poseAt(vision::Timestamp t, bool interpolate = true) const;
                vision::PoseWithCovariance::ConstShPtr motionBetween(vision::Timestamp t0,vision::Timestamp t1, bool interpolate = true) const;
                void append(vision::Timestamp t, vision::PoseWithCovariance::ConstShPtr pose);
                const std::map<vision::Timestamp, vision::PoseWithCovariance::ConstShPtr>& poses() const {return _poses;}
                private:
                
                vision::PoseWithCovariance::ConstShPtr interpolateAt(vision::Timestamp t) const;

                std::map<vision::Timestamp, vision::PoseWithCovariance::ConstShPtr> _poses;

        };
} // namespace pd::vslam::odometry

#endif