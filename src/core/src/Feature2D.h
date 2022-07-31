//
// Created by phil on 30.06.21.
//

#ifndef VSLAM_FEATURE2D_H
#define VSLAM_FEATURE2D_H

#include <memory>
#include <Eigen/Dense>

namespace pd::vslam{

class Point3D;
class FrameRgb;

class Feature2D {
public:
    using ShPtr = std::shared_ptr<Feature2D>;
    using ConstShPtr = std::shared_ptr<Feature2D>;

    Feature2D(const Eigen::Vector2d& position, std::shared_ptr<FrameRgb> frame,std::shared_ptr<Point3D> p3d = nullptr);

    std::shared_ptr<const Point3D> point() const {return _point;}
    std::shared_ptr< Point3D>& point() {return _point;}
    const Eigen::Vector2d& position() const { return _position;}
    std::shared_ptr<FrameRgb>  frame()  { return _frame;}
    std::shared_ptr<const FrameRgb>  frame() const { return _frame;}
    const std::uint64_t& id() const { return _id;}
private:
    const Eigen::Vector2d _position;
    const std::shared_ptr<FrameRgb>  _frame;
    std::shared_ptr< Point3D> _point;
    const std::uint64_t _id;
    static std::uint64_t _idCtr;

};

}


#endif //VSLAM_FEATURE2D_H
