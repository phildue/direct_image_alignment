//
// Created by phil on 30.06.21.
//

#ifndef DIRECT_IMAGE_ALIGNMENT_POINT3D_H
#define DIRECT_IMAGE_ALIGNMENT_POINT3D_H

#include <memory>
#include <vector>
#include "Feature2D.h"

namespace pd{
    namespace vision {

        class Point3D {
        public:
            using ShPtr = std::shared_ptr<Point3D>;
            using ShConstPtr = std::shared_ptr<Point3D>;
        private:
            std::vector<Feature2D::ShConstPtr> _features;
        };

    }}
#endif //DIRECT_IMAGE_ALIGNMENT_POINT_H
