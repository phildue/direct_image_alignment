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

                virtual const Image& rgb() const {return _rgb;}
                virtual const PoseWithCovariance& pose() const {return _pose;}
                virtual const Timestamp& t() const {return _t;}
                virtual Camera::ConstShPtr camera() const { return _cam; }
                virtual ~FrameRgb(){};
                private:
                Image _rgb;
                Camera::ConstShPtr _cam;
                Timestamp _t;
                PoseWithCovariance _pose; //<< Pw = pose * Pf
        };

        class FrameRgbd : public FrameRgb {

                public:

                typedef std::shared_ptr<FrameRgbd> ShPtr;
                typedef std::shared_ptr<const FrameRgbd> ConstShPtr;
                typedef std::unique_ptr<FrameRgbd> UnPtr;
                typedef std::unique_ptr<const FrameRgbd> ConstUnPtr;
                
                FrameRgbd(const Image rgb, const DepthMap& depth, Camera::ConstShPtr cam, const Timestamp& t = 0U, const PoseWithCovariance& pose = {})
                :FrameRgb(rgb,cam,t,pose),_depth(depth){}

                virtual const DepthMap& depth() const {return _depth;}
                private:
                DepthMap _depth;

        };
} // namespace pd::vision



#endif