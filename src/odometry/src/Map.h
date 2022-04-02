#ifndef VSLAM_MAP_H__
#define VSLAM_MAP_H__

#include "core/core.h"
namespace pd::vision{
class Map{
        public:
        typedef std::shared_ptr<Map> ShPtr;
        typedef std::unique_ptr<Map> UnPtr;
        typedef std::shared_ptr<const Map> ConstShPtr;
        typedef std::unique_ptr<const Map> ConstUnPtr;

        virtual void update(FrameRgbd::ConstShPtr frame) {_lastFrame = frame;};
        virtual void updateKf(FrameRgbd::ConstShPtr frame){ _lastKeyFrame = frame;};
        
        FrameRgbd::ConstShPtr lastKf() const { return _lastKeyFrame;}
        FrameRgbd::ConstShPtr lastFrame() const { return _lastFrame;}

        private:
        FrameRgbd::ConstShPtr _lastKeyFrame = nullptr;
        FrameRgbd::ConstShPtr _lastFrame = nullptr;

};

}
#endif// VSLAM_MAP_H__

