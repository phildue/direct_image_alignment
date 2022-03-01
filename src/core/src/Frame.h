#ifndef VSLAM_FRAME_H__
#define VSLAM_FRAME_H__
#include <memory>

#include "types.h"
#include "PoseWithCovariance.h"
namespace pd::vision
{
        
        class FrameRgb{

                public:

                typedef std::shared_ptr<FrameRgb> ShPtr;
                typedef std::shared_ptr<const FrameRgb> ConstShPtr;
                typedef std::unique_ptr<FrameRgb> UnPtr;
                typedef std::unique_ptr<const FrameRgb> ConstUnPtr;
                
                FrameRgb(const Image rgb, Camera::ConstShPtr cam, const Timestamp& t = 0U, const PoseWithCovariance& pose = {})
                :_rgb(rgb),_cam(cam),_t(t),_pose(pose){}

                const Image& rgb() const {return _rgb;}
                const PoseWithCovariance& pose() const {return _pose;}
                const Timestamp& t() const {return _t;}
                Camera::ConstShPtr camera() const { return _cam; }
                private:
                Image _rgb;
                Camera::ConstShPtr _cam;
                Timestamp _t;
                PoseWithCovariance _pose;
        };

        class FrameRgbd{

                public:

                typedef std::shared_ptr<FrameRgbd> ShPtr;
                typedef std::shared_ptr<const FrameRgbd> ConstShPtr;
                typedef std::unique_ptr<FrameRgbd> UnPtr;
                typedef std::unique_ptr<const FrameRgbd> ConstUnPtr;
                
                FrameRgbd(const Image rgb, const DepthMap& depth, Camera::ConstShPtr cam, const Timestamp& t = 0U, const PoseWithCovariance& pose = {})
                :_rgb(rgb),_depth(depth),_cam(cam),_t(t),_pose(pose){}

                const Image& rgb() const {return _rgb;}
                const PoseWithCovariance& pose() const {return _pose;}
                const Timestamp& t() const {return _t;} 
                const DepthMap& depth() const {return _depth;}
                Camera::ConstShPtr camera() const { return _cam; }
                private:
                Image _rgb;
                DepthMap _depth;
                Camera::ConstShPtr _cam;
                Timestamp _t;
                PoseWithCovariance _pose;

        };
} // namespace pd::vision



#endif