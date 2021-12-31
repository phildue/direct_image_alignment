#include <memory>
#include "core/types.h"
#include "WarpAffine.h"
namespace pd{namespace vision{

    WarpAffine::WarpAffine(const Eigen::Vector6d& x, double cx,double cy)
    :_x(x),
    _cx(cx),
    _cy(cy){
        _w = toMat(_x);
    };
    void WarpAffine::updateAdditive(const Eigen::Vector6d& dx)
    {
        _x.noalias() += dx;
        _w = toMat(_x);

    }
    void WarpAffine::updateCompositional(const Eigen::Vector6d& dx)
    {
        _x(0) = _x(0) + dx(0) +_x(0)*dx(0) + _x(2)*dx(1);
        _x(1) = _x(1) + dx(1) +_x(1)*dx(0) + _x(3)*dx(1);
        _x(2) = _x(2) + dx(2) +_x(0)*dx(2) + _x(2)*dx(3);
        _x(3) = _x(3) + dx(3) +_x(1)*dx(2) + _x(3)*dx(3);
        _x(4) = _x(4) + dx(4) +_x(0)*dx(4) + _x(2)*dx(5);
        _x(5) = _x(5) + dx(5) +_x(1)*dx(4) + _x(3)*dx(5);

        _w = toMat(_x);

    }
    Eigen::Vector2d WarpAffine::apply(int u, int v) const { 
        Eigen::Vector3d uv1;
        uv1 << u,v,1;
        auto wuv1 = _w * uv1;
        return {wuv1.x(),wuv1.y()};
    }
    Eigen::Matrix<double,2,6> WarpAffine::J(int u, int v) const {
        Eigen::Matrix<double,2,6> J;
        J << u - _cx,0,v - _cy,0,1,0,
             0,u - _cx,0,v - _cy,0,1;
        return J;
    }
    void WarpAffine::setX(const Eigen::Vector6d& x)
    {
        _x = x;
        _w = toMat(x);
    }
    Eigen::Vector6d WarpAffine::x() const {return _x;}
    Eigen::Matrix3d WarpAffine::toMat(const Eigen::Vector6d& x) const 
    {
        Eigen::Matrix3d w;
        w <<  1+x(0),x(2),x(4),
              x(1),1+x(3),x(5),
              0,0,1;
        return w;
    }

}}