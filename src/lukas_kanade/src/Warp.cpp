#include <memory>
#include "core/core.h"
#include "Warp.h"
namespace pd{namespace vision{

    WarpAffine::WarpAffine(const Eigen::Vector6d& x, double cx,double cy)
    :_x(x),
    _cx(cx),
    _cy(cy){
        _w = toMat(_x);
    }
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

    WarpSE3::WarpSE3(const Eigen::Vector6d& x, const Eigen::MatrixXd& depth, std::shared_ptr<Camera> cam)
    :_x(x)
    ,_pose(Sophus::SE3d::exp(_x))
    ,_depth(depth)
    ,_cam(cam)
    {
    }
    void WarpSE3::updateAdditive(const Eigen::Vector6d& dx)
    {
        _pose = Sophus::SE3d::exp(dx) * _pose;
        _x = _pose.log();
    }
    void WarpSE3::updateCompositional(const Eigen::Vector6d& dx)
    {
        //TODO
        _pose =  Sophus::SE3d::exp(dx)*_pose;
        _x = _pose.log();

    }
    Eigen::Vector2d WarpSE3::apply(int u, int v) const { 
        if (std::isfinite(_depth(v,u)) && _depth(v,u) > 0)
        {
            return _cam->camera2image( _pose * _cam->image2camera({u,v},_depth(v,u)));
        }else{
            return {-1,-1};
        }
    }
    Eigen::Matrix<double,2,WarpSE3::nParameters> WarpSE3::J(int u, int v) const {
        Eigen::Matrix<double,2,6> J = Eigen::Matrix<double,2,6>::Zero();
        if (std::isfinite(_depth(v,u)) && _depth(v,u) > 0)
        {
            const Eigen::Vector3d pCcsRef = _cam->image2camera({u,v},_depth(v,u));
            const Eigen::Matrix<double,2,6> j = _cam->J_xyz2uv(pCcsRef);  
            return j;
        }else{
            return J;
        }
    }
    void WarpSE3::setX(const Eigen::Vector6d& x)
    {
        _x = x;
        _pose = Sophus::SE3d::exp(x);
    }
    Eigen::Vector6d WarpSE3::x() const {return _x;}
    
    const SE3d& WarpSE3::pose() const
    {
        return _pose;
    }

    WarpSE3 WarpSE3::resize(const WarpSE3& w, double scale)
    {
        auto camScaled = std::make_shared<Camera>(*w._cam);
        camScaled->resize(scale);
        return WarpSE3(w._x,algorithm::resize(w._depth,scale),camScaled);
    }

}}