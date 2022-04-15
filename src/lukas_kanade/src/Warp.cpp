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
        _w = _w * toMat(dx);
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


    WarpSE3::WarpSE3(const SE3d& poseCur, const Eigen::MatrixXd& depth, Camera::ConstShPtr camCur, Camera::ConstShPtr camRef, const SE3d& poseRef)
    :_se3(poseCur * poseRef.inverse())
    ,_poseRef(poseRef)
    ,_width(depth.cols())
    ,_camCur(camCur)
    ,_camRef(camRef)
    ,_pcl(depth.rows()*depth.cols())
    {
        _x = _se3.log();
        //TODO move pcl to frame so its only computed once
        for(int v = 0; v < depth.rows(); v++)
        {
            for(int u = 0; u < depth.cols(); u++)
            {
                 /* Exclude pixels that are close to not having depth since we do bilinear interpolation later*/
                 if (std::isfinite(depth(v,u)) && depth(v,u) > 0 &&
                    std::isfinite(depth(v+1,u+1)) && depth(v+1,u+1) > 0  &&
                    std::isfinite(depth(v+1,u-1)) && depth(v+1,u-1) > 0  &&
                    std::isfinite(depth(v-1,u+1)) && depth(v-1,u+1) > 0  &&
                    std::isfinite(depth(v-1,u-1)) && depth(v-1,u-1) > 0
                    )//TODO move to actual interpolation?
                {
                    _pcl[v * _width + u] = _camRef->image2camera({u,v},depth(v,u));
                }else{
                    _pcl[v * _width + u] = Eigen::Vector3d::Zero();
                }
            }
        }
    }
    WarpSE3::WarpSE3(const SE3d& poseCur, const std::vector<Vec3d>& pcl,size_t width, Camera::ConstShPtr camCur, Camera::ConstShPtr camRef, const SE3d& poseRef)
    :_se3(poseCur * poseRef.inverse())
    ,_poseRef(poseRef)
    ,_width(width)
    ,_camCur(camCur)
    ,_camRef(camRef)
    ,_pcl(pcl)
    {_x = _se3.log();}
    void WarpSE3::updateAdditive(const Eigen::Vector6d& dx)
    {
        _se3 = _se3 * Sophus::SE3d::exp(dx);
        _x = _se3.log();
    }
    void WarpSE3::updateCompositional(const Eigen::Vector6d& dx)
    {
        //TODO
        _se3 = _se3 * Sophus::SE3d::exp(dx);
        _x = _se3.log();

    }
    Eigen::Vector2d WarpSE3::apply(int u, int v) const { 
        auto& p = _pcl[ v * _width + u];
        return p.z() > 0.0 ? _camCur->camera2image( _se3 * p) : Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::quiet_NaN());
    }
    Eigen::Matrix<double,2,WarpSE3::nParameters> WarpSE3::J(int u, int v) const {
        /*A tutorial on SE(3) transformation parameterizations and on-manifold optimization 
        A.2. Projection of a point p.43
        Jacobian of uv = K * T_SE3 * p3d
        with respect to tx,ty,tz,rx,ry,rz the parameters of the lie algebra element of T_SE3
        */
        Eigen::Matrix<double,2,6> jac;
        jac.setConstant(std::numeric_limits<double>::quiet_NaN());
        const Eigen::Vector3d& p = _pcl[v * _width + u];
        if (p.z() <= 0.0 ) { return jac;} 

        const double& x = p.x();
        const double& y = p.y();
        const double z_inv = 1./p.z();
        const double z_inv_2 = z_inv*z_inv;

        jac(0,0) = z_inv;              
        jac(0,1) = 0.0;                 
        jac(0,2) = -x*z_inv_2;           
        jac(0,3) = y*jac(0,2);            
        jac(0,4) = 1.0 - x*jac(0,2);   
        jac(0,5) = -y*z_inv;
        

        jac(1,0) = 0.0;                 
        jac(1,1) = z_inv;             
        jac(1,2) = -y*z_inv_2;           
        jac(1,3) = -1.0 + y*jac(1,2);      
        jac(1,4) = -jac(0,3);            
        jac(1,5) = x*z_inv;    
        jac.row(0) *= _camRef->fx();     
        jac.row(1) *= _camRef->fy();
        return jac;
        
    }

    Image WarpSE3::apply(const Image& img) const
    {
        Image warped = Image::Zero(img.rows(),img.cols());  
        for(int i = 0; i < warped.rows(); i++)
        {
            for(int j = 0; j < warped.cols(); j++)
            {
                 Eigen::Vector2d uvI = apply(j,i);
                if (1 < uvI.x() && uvI.x() < img.cols() - 1  &&
                    1 < uvI.y() && uvI.y() < img.rows() - 1)
                {
                    warped(i,j) =  algorithm::bilinearInterpolation(img,uvI.x(),uvI.y());
                }
            }
        }
        return warped;
    }

    DepthMap WarpSE3::apply(const DepthMap& img) const
    {
        DepthMap warped = DepthMap::Zero(img.rows(),img.cols());  
        for(int i = 0; i < warped.rows(); i++)
        {
            for(int j = 0; j < warped.cols(); j++)
            {
                Eigen::Vector2d uvI = apply(j,i);
                if (1 < uvI.x() && uvI.x() < img.cols() - 1  &&
                    1 < uvI.y() && uvI.y() < img.rows() - 1 
                    )
                { //TODO check for invalid
                    warped(i,j) =  algorithm::bilinearInterpolation(img,uvI.x(),uvI.y());
                }
            }
        }
        return warped;
    }


    void WarpSE3::setX(const Eigen::Vector6d& x)
    {
        _x = x;
        _se3 = Sophus::SE3d::exp(x);
    }
    Eigen::Vector6d WarpSE3::x() const {return _x;}
    
    SE3d WarpSE3::poseCur() const
    {
        return _se3 * _poseRef;
    }

   

}}