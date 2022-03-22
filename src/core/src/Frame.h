#ifndef VSLAM_FRAME_H__
#define VSLAM_FRAME_H__
#include <memory>

#include "types.h"
#include "PoseWithCovariance.h"
#include "algorithm.h"
#include "Camera.h"
#include "Kernel2d.h"
namespace pd::vision
{
        
        class FrameRgb{

                public:

                typedef std::shared_ptr<FrameRgb> ShPtr;
                typedef std::shared_ptr<const FrameRgb> ConstShPtr;
                typedef std::unique_ptr<FrameRgb> UnPtr;
                typedef std::unique_ptr<const FrameRgb> ConstUnPtr;
                
                FrameRgb(const Image intensity, Camera::ConstShPtr cam, const Timestamp& t = 0U, const PoseWithCovariance& pose = {})
                :_intensity(intensity),
                _dIx(algorithm::conv2d(intensity.cast<double>(),Kernel2d<double>::scharrX()).cast<int>()),
                _dIy(algorithm::conv2d(intensity.cast<double>(),Kernel2d<double>::scharrY()).cast<int>()),
                _cam(cam),_t(t),_pose(pose){}

                virtual const Image& intensity() const {return _intensity;}
                virtual const MatXi& dIx() const { return _dIx;}
                virtual const MatXi& dIy() const { return _dIy;}

                virtual const PoseWithCovariance& pose() const {return _pose;}
                virtual const Timestamp& t() const {return _t;}
                virtual Camera::ConstShPtr camera() const { return _cam; }
                virtual ~FrameRgb(){};
                private:
                Image _intensity;
                MatXi _dIx,_dIy;
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
        class RgbdPyramid {
                public:
                typedef std::shared_ptr<RgbdPyramid> ShPtr;
                typedef std::shared_ptr<const RgbdPyramid> ConstShPtr;
                typedef std::unique_ptr<RgbdPyramid> UnPtr;
                typedef std::unique_ptr<const RgbdPyramid> ConstUnPtr;
                
                RgbdPyramid(const Image rgb, const DepthMap& depth, Camera::ConstShPtr cam,const std::vector<double>& scales, const Timestamp& t = 0U, const PoseWithCovariance& pose = {});
                RgbdPyramid(const Image rgb, const DepthMap& depth, Camera::ConstShPtr cam,const uint32_t levels, const Timestamp& t = 0U, const PoseWithCovariance& pose = {});
                virtual const DepthMap& depth(size_t level = 0) const {return _levels[level]->depth();}
                virtual const Image& intensity(size_t level = 0) const {return _levels[level]->intensity();}
                virtual const MatXi& dIx(size_t level = 0) const { return _levels[level]->dIx();}
                virtual const MatXi& dIy(size_t level = 0) const { return _levels[level]->dIy();}
                virtual const PoseWithCovariance& pose() const {return _pose;}
                virtual const Timestamp& t() const {return _t;}
                virtual Camera::ConstShPtr camera(size_t level = 0) const { return _levels[level]->camera(); }
                const std::vector<double>& scales() const { return _scales;}
                size_t nLevels() const { return _scales.size();}
                private:
                std::vector<FrameRgbd::ShPtr> _levels;
                std::vector<double> _scales;
                Timestamp _t;
                PoseWithCovariance _pose; //<< Pw = pose * Pf

        };
} // namespace pd::vision



#endif