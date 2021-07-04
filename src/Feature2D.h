//
// Created by phil on 30.06.21.
//

#ifndef DIRECT_IMAGE_ALIGNMENT_FEATURE2D_H
#define DIRECT_IMAGE_ALIGNMENT_FEATURE2D_H

#include <memory>
#include <Eigen/Dense>

namespace pd{
    namespace vision{
class Point3D;
class Feature2D {
public:
    using ShPtr = std::shared_ptr<Feature2D>;
    using ShConstPtr = std::shared_ptr<Feature2D>;

    std::shared_ptr<const Point3D> point() const {return _point;}
    const Eigen::Vector2d& position() const { return _position;}
private:
    std::shared_ptr<const Point3D> _point;
    Eigen::Vector2d _position;
};

    }
}

#endif //DIRECT_IMAGE_ALIGNMENT_FEATURE2D_H
