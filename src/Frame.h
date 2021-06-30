//
// Created by phil on 30.06.21.
//

#ifndef DIRECT_IMAGE_ALIGNMENT_FRAME_H
#define DIRECT_IMAGE_ALIGNMENT_FRAME_H

#include <memory>
#include <vector>

#include "Feature2D.h"

namespace pd{
    namespace vision {

        class Frame {
        public:
            using ShPtr = std::shared_ptr<Frame>;
            using ShConstPtr = std::shared_ptr<Frame>;

        private:
            std::vector<Feature2D::ShConstPtr> _features;
        };
    }}

#endif //MYLIBRARY_FRAME_H
