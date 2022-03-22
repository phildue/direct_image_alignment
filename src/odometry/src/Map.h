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

        virtual void update(RgbdPyramid::ConstShPtr UNUSED(frame)) {};
        virtual void updateKf(RgbdPyramid::ConstShPtr frame){ _lastKeyFrame = frame;};
        
        RgbdPyramid::ConstShPtr lastKf() const { return _lastKeyFrame;}

        private:
        RgbdPyramid::ConstShPtr _lastKeyFrame = nullptr;
};

}
#endif// VSLAM_MAP_H__

