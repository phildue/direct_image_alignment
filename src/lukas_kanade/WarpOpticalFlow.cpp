#include <memory>
#include "core/types.h"
#include "WarpOpticalFlow.h"
namespace pd{namespace vision{
    WarpOpticalFlow::WarpOpticalFlow(const Eigen::Vector2d& x)
    :_x(x){
        _w = toMat(_x);
    };
    void WarpOpticalFlow::updateAdditive(const Eigen::Vector2d& dx)
    {
        _x.noalias() += dx;
        _w = toMat(_x);

    }
    void WarpOpticalFlow::updateCompositional(const Eigen::Vector2d& dx)
    {
        //TODO
        _w = toMat(dx) * _w;
        _x(0) = _w(0,2);
        _x(1) = _w(1,2);

    }
    Eigen::Vector2d WarpOpticalFlow::apply(int u, int v) const { 
        Eigen::Vector2d uv;
        uv << u,v;
        Eigen::Vector2d wuv = uv + _x;
        return wuv;
    }
    Eigen::Matrix<double,2,WarpOpticalFlow::nParameters> WarpOpticalFlow::J(int u, int v) const {
        return Eigen::Matrix<double,2,nParameters>::Identity();
    }
    void WarpOpticalFlow::setX(const Eigen::Vector2d& x)
    {
        _x = x;
        _w = toMat(x);
    }
    Eigen::Matrix3d WarpOpticalFlow::toMat(const Eigen::Vector2d& x) const 
    {
        Eigen::Matrix3d w;
        w <<  1,0,x(0),
              0,1,x(1),
              0,0,1;
        return w;
    }


}}