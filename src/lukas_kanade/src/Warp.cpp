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
    }
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
    Eigen::Matrix<double,2,WarpOpticalFlow::nParameters> WarpOpticalFlow::J(int UNUSED(u), int UNUSED(v)) const {
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


    WarpSE3::WarpSE3(const SE3d& world2img, const Eigen::MatrixXd& depth, Camera::ConstShPtr camImg, Camera::ConstShPtr camTempl, const SE3d& templ2world)
    :_x(world2img.log())
    ,_world2img(world2img)
    ,_templ2world(templ2world)
    ,_depth(depth)
    ,_camImg(camImg)
    ,_camTempl(camTempl)
    {
    }

    void WarpSE3::updateAdditive(const Eigen::Vector6d& dx)
    {
        _world2img = Sophus::SE3d::exp(dx) * _world2img;
        _x = _world2img.log();
    }
    void WarpSE3::updateCompositional(const Eigen::Vector6d& dx)
    {
        //TODO
        _world2img =  Sophus::SE3d::exp(dx)*_world2img;
        _x = _world2img.log();

    }
    Eigen::Vector2d WarpSE3::apply(int u, int v) const { 
        if (std::isfinite(_depth(v,u)) && _depth(v,u) > 0)
        {
            return _camImg->camera2image( _world2img * _templ2world *_camTempl->image2camera({u,v},_depth(v,u)));
        }else{
            return {-1,-1};
        }
    }
    Eigen::Matrix<double,2,WarpSE3::nParameters> WarpSE3::J(int u, int v) const {
        //A tutorial on SE(3) transformation parameterizations and on-manifold optimization 
        //A.2. Projection of a point p.43
        Eigen::Matrix<double,2,6> jac = Eigen::Matrix<double,2,6>::Zero();
        if (std::isfinite(_depth(v,u)) && _depth(v,u) > 0)
        {
            //TODO should this be pRef or pTarget?
            const Eigen::Vector3d pRef = _camTempl->image2camera({u,v},_depth(v,u));
            const double& x = pRef.x();
            const double& y = pRef.y();
            const double z_inv = 1./pRef.z();
            const double z_inv_2 = z_inv*z_inv;

            jac(0,0) = z_inv;              
            jac(0,1) = 0.0;                 
            jac(0,2) = -x*z_inv_2;           
            jac(0,3) = -y*jac(0,2);            
            jac(0,4) = (1.0 + x*jac(0,2));   
            jac(0,5) = -y*z_inv;
            jac.row(0) *= _camTempl->fx();             

            jac(1,0) = 0.0;                 
            jac(1,1) = z_inv;             
            jac(1,2) = -y*z_inv_2;           
            jac(1,3) = -(1.0 + y*jac(1,2));      
            jac(1,4) = jac(0,3);            
            jac(1,5) = x*z_inv;    
            jac.row(1) *= _camTempl->fy();
        }
        return jac;
        
    }
    void WarpSE3::setX(const Eigen::Vector6d& x)
    {
        _x = x;
        _world2img = Sophus::SE3d::exp(x);
    }
    Eigen::Vector6d WarpSE3::x() const {return _x;}
    
    SE3d WarpSE3::SE3() const
    {
        return _world2img;
    }

   

}}