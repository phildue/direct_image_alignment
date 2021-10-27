#ifndef VSLAM_LUKAS_KANADE_H__
#define VSLAM_LUKAS_KANADE_H__
#include "core/types.h"
#include "solver/solver.h"
#include <memory>
namespace pd{namespace vision{

template<int nParameters>
class LukasKanade{
public:
    LukasKanade (const Image& templ, const Image& image, int maxIterations, double minStepSize = 1e-3, double minGradient = 1e-3);
    void solve(Eigen::Matrix<double,nParameters,1>& x) const;

protected:

    //
    // r = T(x) - I(W(x,p))
    //
    bool computeResidual(const Eigen::Matrix<double,nParameters,1>& x, Eigen::VectorXd& r, Eigen::VectorXd& w) const;
   
    //
    // J = Ixy*dW/dp
    //
    bool computeJacobian(const Eigen::Matrix<double,nParameters,1>& x, Eigen::Matrix<double,-1,nParameters>& j) const;

    virtual bool updateX(const Eigen::Matrix<double,nParameters,1>& dx, Eigen::Matrix<double,nParameters,1>& x) const = 0;

    virtual Eigen::MatrixXi warp(const Eigen::Matrix<double,nParameters,1>& x, const Eigen::MatrixXi& img) const = 0;
    virtual Image warp(const Eigen::Matrix<double,nParameters,1>& x, const Image& img) const = 0;

    virtual Eigen::Matrix<double,2,nParameters> jacobianWarp(int v, int u) const = 0;

    const Image _T;
    const Image _Iref;
    Eigen::MatrixXi _dIx;
    Eigen::MatrixXi _dIy;
    std::shared_ptr<const Solver<nParameters>> _solver;
};
}}
#include "LukasKanade.hpp"
#endif
