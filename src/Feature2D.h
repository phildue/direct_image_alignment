//
// Created by phil on 30.06.21.
//

#ifndef DIRECT_IMAGE_ALIGNMENT_FEATURE2D_H
#define DIRECT_IMAGE_ALIGNMENT_FEATURE2D_H

#include <memory>
#include "Point3D.h"

namespace pd{
    namespace vision{
class Point3D;
class Feature2D {
public:
    using ShPtr = std::shared_ptr<Feature2D>;
    using ShConstPtr = std::shared_ptr<Feature2D>;

    Point3D::ShConstPtr point() const {return _point;}
private:
    Point3D::ShConstPtr _point;
};

    }
}

#endif //DIRECT_IMAGE_ALIGNMENT_FEATURE2D_H
