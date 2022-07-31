#ifndef VSLAM_MAP_H__
#define VSLAM_MAP_H__
#include <deque>
#include "core/core.h"
namespace pd::vslam{
class Map{
        public:
        typedef std::shared_ptr<Map> ShPtr;
        typedef std::unique_ptr<Map> UnPtr;
        typedef std::shared_ptr<const Map> ConstShPtr;
        typedef std::unique_ptr<const Map> ConstUnPtr;

        Map();

        virtual void update(FrameRgbd::ShPtr frame, bool isKeyFrame);
        virtual void update(const std::vector<Point3D::ShPtr>& points);
        
        FrameRgbd::ConstShPtr lastKf(size_t idx = 0) const { return _keyFrames.size() <= idx ? nullptr : _keyFrames.at(idx);}
        FrameRgbd::ConstShPtr lastFrame(size_t idx = 0) const { return _frames.size() <= idx ? nullptr : _frames.at(idx);}

        const std::deque<FrameRgbd::ShPtr>& keyFrames() { return _keyFrames;}
        const std::deque<FrameRgbd::ShPtr>& frames() { return _frames;}
        const std::vector<Point3D::ShPtr>& points() {return _points;}
      
        private:
        std::deque<FrameRgbd::ShPtr> _frames;
        std::deque<FrameRgbd::ShPtr> _keyFrames;
        std::vector<Point3D::ShPtr> _points;
        const size_t _maxFrames,_maxKeyFrames;

};

}
#endif// VSLAM_MAP_H__

