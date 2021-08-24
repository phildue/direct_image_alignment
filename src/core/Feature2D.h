//
// Created by phil on 30.06.21.
//

#ifndef DIRECT_IMAGE_ALIGNMENT_FEATURE2D_H
#define DIRECT_IMAGE_ALIGNMENT_FEATURE2D_H

#include <memory>
#include <Eigen/Dense>
#include "feature_extraction/Descriptor.h"

namespace pd{
    namespace vision{
class Point3D;
class Frame;



class Feature2D {
public:
    using ShPtr = std::shared_ptr<Feature2D>;
    using ShConstPtr = std::shared_ptr<Feature2D>;

    Feature2D(const Eigen::Vector2d& position,std::shared_ptr<Descriptor> descriptor, std::shared_ptr<Frame> frame,std::shared_ptr<Point3D> p3d = nullptr, int level = 0);

    std::shared_ptr<const Point3D> point() const {return _point;}
    std::shared_ptr< Point3D>& point() {return _point;}
    const Eigen::Vector2d& position() const { return _position;}
    std::shared_ptr<Frame>  frame()  { return _frame;}
    std::shared_ptr<const Frame>  frame() const { return _frame;}
    const std::uint64_t& id() const { return _id;}
    std::shared_ptr<const Descriptor> descriptor() const { return _descriptor;}
private:
    std::shared_ptr< Point3D> _point;
    const Eigen::Vector2d _position;
    const std::shared_ptr< Frame>  _frame;
    const std::uint64_t _id;
    const std::shared_ptr<Descriptor> _descriptor;
    const int _level;
    static std::uint64_t _idCtr;


};

    }
}

#endif //DIRECT_IMAGE_ALIGNMENT_FEATURE2D_H
