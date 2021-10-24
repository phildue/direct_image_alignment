#ifndef LUKAS_KANADE_H__
#define LUKAS_KANADE_H__
#include "core/types.h"
#include "solver/solver.h"
namespace pd{namespace vision{
class LukasKanadeOpticalFlow{
public:
    LukasKanade (const Image& templ, const Image& image, int maxIterations, double minStepSize = 1e-3, double minGradient = 1e-3);
   
    //
    // r = T(x) - I(W(x,p))
    //
    bool computeResidual(const Eigen::Vector2d& x, Eigen::VectorXd& r, Eigen::VectorXd& w) const;
   
    //
    // J = Ixy*dW/dp
    //
    bool computeJacobian(const Eigen::Vector2d& x, Eigen::Matrix<double,-1,2>d& j) const;

    bool updateX(const Eigen::Vector2d& dx, Eigen::VectorXd& x);
protected:
    const Image _T;
    const Image _Iref;
    Eigen::MatrixXi _dIx;
    Eigen::MatrixXi _dIy;
    std::shared_ptr<const Solver<2>> _solver;
    
};
class LukasKanadeAffine{
public:
    LukasKanade (const Image& templ, const Image& image);
   
    //
    // r = T(x) - I(W(x,p))
    //
    bool computeResidual(const Eigen::VectorXd& x, Eigen::VectorXd& r, Eigen::VectorXd& w) const;
   
    //
    // J = Ixy*dW/dp
    //
    bool computeJacobian(const Eigen::VectorXd& x, Eigen::MatrixXd& j) const;

    bool updateX(const Eigen::VectorXd& dx, Eigen::VectorXd& x) const;
protected:
    const Image _T;
    const Image _Iref;
    Eigen::MatrixXi _dIx;
    Eigen::MatrixXi _dIy;
    LevenbergMarquardt _solver;
    
};
}}
#endif