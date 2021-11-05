#include "LukasKanadeSE3d.h"
#include "solver/LevenbergMarquardt.h"
#include "utils/visuals.h"
#include "utils/utils.h"
namespace pd{namespace vision{

    LukasKanadeSE3d::LukasKanadeSE3d (const Image& templ, const Image& image,const Eigen::MatrixXd& depth, std::shared_ptr<Camera> cam, int maxIterations, double minStepSize, double minGradient)
    : LukasKanade<6>(templ,image,maxIterations,minStepSize,minGradient)
    , _depth(depth)
    , _cam(cam)
    {}

    void LukasKanadeSE3d::solve(Eigen::Vector6d& x) 
    {
        _pose = Sophus::SE3d::exp(x);
        return _solver->solve(x);
    }

    bool LukasKanadeSE3d::updateX(const Eigen::Vector6d& dx, Eigen::Vector6d& x)
    {
        _pose = Sophus::SE3d::exp(x)*Sophus::SE3d::exp(dx);
        x.noalias() = _pose.log();

        return true;
    }

    
    Eigen::Matrix<double,2,6> LukasKanadeSE3d::jacobianWarp(int v, int u)
    {
        Eigen::Matrix<double,2,6> J = Eigen::Matrix<double,2,6>::Zero();
        if (std::isfinite(_depth(v,u)) && _depth(v,u) > 0)
        {
            const Eigen::Vector3d pCcsRef = _cam->image2camera({u,v},_depth(v,u));
            //const Eigen::Vector2d uvRef = _cam->camera2image(pCcsRef);
            const Eigen::Matrix<double,2,6> j = _cam->J_xyz2uv(pCcsRef);  
            utils::throw_if_nan(j.cast<double>(),"J");
            return j;
        }else{
            return J;
        }
        
        
    }

    Eigen::Vector2d LukasKanadeSE3d::warp(int u, int v,const Eigen::Vector6d& x) const
    {
         return _cam->camera2image( Sophus::SE3d::exp(x) * _cam->image2camera({u,v},_depth(v,u)));
    }


}}