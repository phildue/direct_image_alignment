//
// Created by phil on 30.06.21.
//

#ifndef DIRECT_IMAGE_ALIGNMENT_POSE_H
#define DIRECT_IMAGE_ALIGNMENT_POSE_H

#include <memory>

namespace pd{
    namespace vision {


        class Pose {
        public:
            using ShPtr = std::shared_ptr<Pose>;
            using ShConstPtr = std::shared_ptr<Pose>;

        };
    }}

#endif //DIRECT_IMAGE_ALIGNMENT_POSE_H
