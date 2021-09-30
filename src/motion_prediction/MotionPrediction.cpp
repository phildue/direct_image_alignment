//
// Created by phil on 29.09.21.
//

#include <vslam/vslam.h>
#include "MotionPrediction.h"
#include "core/algorithm.h"

namespace pd{namespace vision {

        void MotionPrediction::update(const Sophus::SE3d &pose, Timestamp t) {

            if(_lastPoseT > 0)
            {
                _ready = true;
            }
            if (_ready)
            {
                const auto dt = (t - _lastPoseT);
                if ( dt <= 0 )
                {
                    throw pd::Exception("dt is 0.");

                }
                _speeds = algorithm::computeRelativeTransform(_lastPose,pose).log()/dt;
            }
            _lastPose = pose;
            _lastPoseT = t;
        }

        Sophus::SE3d MotionPrediction::predict(Timestamp t) const{
            if (!_ready)
            {
                throw pd::Exception("Predictor not ready.");
            }
            return Sophus::SE3d::exp((t - _lastPoseT) *_speeds)*_lastPose;
        }
    }}