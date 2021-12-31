#include <memory>
#include "core/types.h"
#include "core/Camera.h"
#include "WarpSE3.h"
#include "utils/utils.h"
namespace pd{namespace vision{
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
        if (std::isinf(_depth(v,u)) || _depth(v,u) <= 0 || std::isnan(_depth(v,u)))
        {
            return {-1,-1};
        }
        return _cam->camera2image( _pose * _cam->image2camera({u,v},_depth(v,u)));
    }
    Eigen::Matrix<double,2,WarpSE3::nParameters> WarpSE3::J(int u, int v) const {
        Eigen::Matrix<double,2,6> J = Eigen::Matrix<double,2,6>::Zero();
        if (std::isfinite(_depth(v,u)) && _depth(v,u) > 0)
        {
            const Eigen::Vector3d pCcsRef = _cam->image2camera({u,v},_depth(v,u));
            const Eigen::Matrix<double,2,6> j = _cam->J_xyz2uv(pCcsRef);  
            utils::throw_if_nan(j.cast<double>(),"J");
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

}}