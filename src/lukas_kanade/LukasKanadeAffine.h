#ifndef VSLAM_LUKAS_KANADE_AFFINE_H__
#define VSLAM_LUKAS_KANADE_AFFINE_H__
#include <memory>
#include "core/types.h"
#include "solver/solver.h"
#include "LukasKanade.h"
namespace pd{namespace vision{
class LukasKanadeAffine : public LukasKanade<6>{
public:
    LukasKanadeAffine (const Image& templ, const Image& image, int maxIterations, double minStepSize = 1e-3, double minGradient = 1e-3);
    void solve(Eigen::Vector6d& x) const;

protected:

    //
    // r = T(x) - I(W(x,p))
    //
    bool updateX(const Eigen::Vector6d& dx, Eigen::Vector6d& x) const override;

    Eigen::MatrixXi warp(const Eigen::Matrix<double,6,1>& x, const Eigen::MatrixXi& img) const override;
    void warp(const Eigen::Matrix<double,6,1>& x, const Image& img, Image& out, Eigen::MatrixXd& mask) const override;

    Eigen::Matrix<double,2,6> jacobianWarp(int v, int u) const override;


    const double _cx,_cy;

    
};
}}
#endif