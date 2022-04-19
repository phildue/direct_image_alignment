#ifndef VSLAM_MAP_H__
#define VSLAM_MAP_H__
#include <deque>
#include "core/core.h"
namespace pd::vision{
class Map{
        public:
        typedef std::shared_ptr<Map> ShPtr;
        typedef std::unique_ptr<Map> UnPtr;
        typedef std::shared_ptr<const Map> ConstShPtr;
        typedef std::unique_ptr<const Map> ConstUnPtr;

        Map();

        virtual void update(FrameRgbd::ConstShPtr frame, bool isKeyFrame);
        
        FrameRgbd::ConstShPtr lastKf(size_t idx = 0) const { return _keyFrames.size() <= idx ? nullptr : _keyFrames.at(idx);}
        FrameRgbd::ConstShPtr lastFrame(size_t idx = 0) const { return _frames.size() <= idx ? nullptr : _frames.at(idx);}

        const std::deque<FrameRgbd::ConstShPtr>& keyFrames() const { return _keyFrames;};
        const std::deque<FrameRgbd::ConstShPtr>& frames() const { return _frames;};
        private:
        std::deque<FrameRgbd::ConstShPtr> _keyFrames;
        std::deque<FrameRgbd::ConstShPtr> _frames;
        size_t _maxFrames = 7,_maxKeyFrames = 7;

};

}
#endif// VSLAM_MAP_H__

