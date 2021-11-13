#ifndef VSLAM_LUKAS_KANADE_AFFINE_H__
#define VSLAM_LUKAS_KANADE_AFFINE_H__
#include <memory>
#include "core/types.h"
#include "solver/solver.h"
#include "LukasKanade.h"
namespace pd{namespace vision{

class WarpAffine
{
    public:
    inline constexpr static int nParameters = 6;
    WarpAffine(const Eigen::Vector6d& x, double cx,double cy)
    :_x(x),
    _cx(cx),
    _cy(cy){
        _w = toMat(_x);
    };
    void updateAdditive(const Eigen::Vector6d& dx)
    {
        _x.noalias() += dx;
        _w = toMat(_x);

    }
    void updateCompositional(const Eigen::Vector6d& dx)
    {
        //TODO
        _w = toMat(dx) * _w;
        _x(0) = _w(0,0) - 1;
        _x(1) = _w(1,0);
        _x(2) = _w(0,1);
        _x(3) = _w(1,1) - 1;
        _x(4) = _w(0,2);
        _x(5) = _w(1,2);

    }
    Eigen::Vector2d apply(int u, int v) const { 
        Eigen::Vector3d uv1;
        uv1 << u,v,1;
        auto wuv1 = _w * uv1;
        return {wuv1.x(),wuv1.y()};
    }
    Eigen::Matrix<double,2,6> J(int u, int v) const {
        Eigen::Matrix<double,2,6> J;
        J << u - _cx,0,v - _cy,0,1,0,
             0,u - _cx,0,v - _cy,0,1;
        return J;
    }
    void setX(const Eigen::Vector6d& x)
    {
        _x = x;
        _w = toMat(x);
    }
    Eigen::Vector6d x() const {return _x;}
    private:
    Eigen::Matrix3d toMat(const Eigen::Vector6d& x) const 
    {
        Eigen::Matrix3d w;
        w <<  1+x(0),x(2),x(4),
              x(1),1+x(3),x(5),
              0,0,1;
        return w;
    }
    Eigen::Matrix3d _w;
    Eigen::Vector6d _x;
    const double _cx,_cy;
};

typedef LukasKanade<WarpAffine> LukasKanadeAffine;
}}
#endif