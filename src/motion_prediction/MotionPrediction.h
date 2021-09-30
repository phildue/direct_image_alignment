//
// Created by phil on 29.09.21.
//

#ifndef VSLAM_MOTIONPREDICTION_H
#define VSLAM_MOTIONPREDICTION_H

#include <sophus/se3.hpp>
#include "core/types.h"

namespace pd{namespace vision {


        class MotionPrediction {
        public:
            void update(const Sophus::SE3d& pose, Timestamp t);
            Sophus::SE3d predict(Timestamp t) const;
        private:
            Eigen::Vector6d _speeds;
            Sophus::SE3d _lastPose;
            Timestamp _lastPoseT = 0U;
            bool _ready = false;
        };

    }}
#endif //VSLAM_MOTIONPREDICTION_H
