#include "Map.h"
namespace pd::vision{

        Map::Map()
        :_frames()
        ,_keyFrames()
        {}
        void Map::update(FrameRgbd::ConstShPtr frame, bool isKeyFrame)
        {
                if(_frames.size() >= _maxFrames){
                        _frames.pop_back();
                }
                _frames.push_front( frame );

                
                if (isKeyFrame){

                        if(_keyFrames.size() >= _maxKeyFrames){
                                _keyFrames.pop_back();
                        }
                        _keyFrames.push_front( frame );

                }
        }
}