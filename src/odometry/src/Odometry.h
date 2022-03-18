#ifndef VSLAM_ODOMETRY
#define VSLAM_ODOMETRY

#include "core/core.h"
namespace pd::vision{
class Odometry{
        public:
        typedef std::shared_ptr<Odometry> ShPtr;
        typedef std::unique_ptr<Odometry> UnPtr;
        typedef std::shared_ptr<const Odometry> ConstShPtr;
        typedef std::unique_ptr<const Odometry> ConstUnPtr;

        virtual void update(FrameRgbd::ConstShPtr frame) = 0;
        
        virtual PoseWithCovariance::ConstShPtr pose() const = 0;
        virtual PoseWithCovariance::ConstShPtr speed() const = 0;
        
        static ShPtr make();

};

}
#endif// VSLAM_ODOMETRY

