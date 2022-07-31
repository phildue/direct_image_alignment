#include "Map.h"
namespace pd::vslam{

        Map::Map()
        :_frames()
        ,_keyFrames()
        ,_maxFrames(7)
        ,_maxKeyFrames(7)
        {}
        void Map::update(FrameRgbd::ShPtr frame, bool isKeyFrame)
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
        void Map::update(const std::vector<Point3D::ShPtr>& points)
        {
                _points = points;
        }

}