#include "LukasKanadeAffine.h"
#include "solver/LevenbergMarquardt.h"
#include "utils/visuals.h"

namespace pd{namespace vision{

    LukasKanadeAffine::LukasKanadeAffine (const Image& templ, const Image& image, int maxIterations, double minStepSize, double minGradient)
    : LukasKanade<6>(templ,image,maxIterations,minStepSize,minGradient)
    , _cx(_T.cols()/2)
    , _cy(_T.rows()/2)
    {
      
    }

     void LukasKanadeAffine::solve(Eigen::Vector6d& x) 
    {
        return _solver->solve(x);
    }
    //
    // J = Ixy*dW/dp
    //
    bool LukasKanadeAffine::updateX(const Eigen::Vector6d& dx, Eigen::Vector6d& x)
    {
        x.noalias() += dx;

        return true;
    }

    Eigen::Matrix<double,2,6> LukasKanadeAffine::jacobianWarp(int v, int u)
    {
        Eigen::Matrix<double,2,6> Jwarp;
            Jwarp << u - _cx,0,v - _cy,0,1,0,
                     0,u - _cx,0,v - _cy,0,1;
        return Jwarp;
    }
    Eigen::Vector2d LukasKanadeAffine::warp(int u, int v,const Eigen::Vector6d& x) const
    {
        Eigen::Matrix3d warp;
        warp << 1+x(0),   x(2), x(4),
                x(1), 1+x(3), x(5),
                0,      0,    1; 
        const Eigen::Vector3d xy1(u - _cx,v - _cy,1);
        const Eigen::Vector3d xy1Ref = warp * xy1;
        Eigen::Vector2d uvRef;
        uvRef << xy1Ref.x() + _cx,xy1Ref.y() + _cy;
        return uvRef;
              
    }

}}