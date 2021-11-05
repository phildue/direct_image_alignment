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
    void solve(Eigen::Vector6d& x);

protected:

    //
    // r = T(x) - I(W(x,p))
    //
    bool updateX(const Eigen::Vector6d& dx, Eigen::Vector6d& x) override;

    Eigen::Matrix<double,2,6> jacobianWarp(int v, int u) override;

    Eigen::Vector2d warp(int u, int v,const Eigen::Vector6d& x) const override;

    const double _cx,_cy;

    
};
}}
#endif