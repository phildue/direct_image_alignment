#ifndef VSLAM_LUKAS_KANADE_AFFINE_H__
#define VSLAM_LUKAS_KANADE_AFFINE_H__
#include <memory>
#include "core/types.h"
#include "core/Camera.h"
#include "solver/solver.h"
namespace pd{namespace vision{
class LukasKanadeSE3d{
public:
    LukasKanadeSE3d (const Image& templ, const Image& image, const Eigen::MatrixXd& depth, std::shared_ptr<Camera> cam, int maxIterations, double minStepSize = 1e-3, double minGradient = 1e-3);
    void solve(Eigen::Vector6d& x) const;

protected:

    //
    // r = T(x) - I(W(x,p))
    //
    bool computeResidual(const Eigen::Vector6d& x, Eigen::VectorXd& r, Eigen::VectorXd& w) const;
   
    //
    // J = Ixy*dW/dp
    //
    bool computeJacobian(const Eigen::Vector6d& x, Eigen::Matrix<double,-1,6>& j) const;

    bool updateX(const Eigen::Vector6d& dx, Eigen::Vector6d& x) const;

    const Image _T;
    const Image _Iref;
    const Eigen::MatrixXd _depth;
    Eigen::MatrixXi _dIx;
    Eigen::MatrixXi _dIy;

    const std::shared_ptr<const Camera> _cam;
    const std::shared_ptr<const Solver<6>> _solver;
    
};
}}
#endif