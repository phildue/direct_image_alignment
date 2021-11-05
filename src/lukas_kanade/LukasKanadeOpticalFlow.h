#ifndef VSLAM_LUKAS_KANADE_OPTICAL_FLOW_H__
#define VSLAM_LUKAS_KANADE_OPTICAL_FLOW_H__
#include <memory>
#include "core/types.h"
#include "solver/solver.h"
namespace pd{namespace vision{
class LukasKanadeOpticalFlow{
public:
    LukasKanadeOpticalFlow (const Image& templ, const Image& image, int maxIterations, double minStepSize = 1e-3, double minGradient = 1e-3);
   
    void solve(Eigen::Vector2d& x) const;
   
protected:
    //
    // r = T(x) - I(W(x,p))
    //
    bool computeResidual(const Eigen::Vector2d& x, Eigen::VectorXd& r, Eigen::VectorXd& w) const;
   
    //
    // J = Ixy*dW/dp
    //
    bool computeJacobian(const Eigen::Vector2d& x, Eigen::Matrix<double,-1,2>& j) const;

    bool updateX(const Eigen::Vector2d& dx, Eigen::Vector2d& x) const;

    Eigen::Vector2d warp(int u, int v,const Eigen::Vector2d& x) const;


    const Image _T;
    const Image _Iref;
    Eigen::MatrixXi _dIx;
    Eigen::MatrixXi _dIy;
    const std::shared_ptr<const Solver<2>> _solver;
    
};

}}
#endif