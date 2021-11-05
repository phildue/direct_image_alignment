#ifndef VSLAM_LUKAS_KANADE_AFFINE_H__
#define VSLAM_LUKAS_KANADE_AFFINE_H__
#include <memory>
#include "core/types.h"
#include "core/Camera.h"
#include "solver/solver.h"
#include "LukasKanade.h"
namespace pd{namespace vision{
class LukasKanadeSE3d : public LukasKanade<6>{
public:
    LukasKanadeSE3d (const Image& templ, const Image& image, const Eigen::MatrixXd& depth, std::shared_ptr<Camera> cam, int maxIterations, double minStepSize = 1e-3, double minGradient = 1e-3);
    void solve(Eigen::Vector6d& x);

protected:


    bool updateX(const Eigen::Vector6d& dx, Eigen::Vector6d& x) override;

    Eigen::Matrix<double,2,6> jacobianWarp(int v, int u) override;

    Eigen::Vector2d warp(int u, int v,const Eigen::Vector6d& x) const override;

    Eigen::MatrixXd _depth;
    std::shared_ptr<Camera> _cam;
    SE3d _pose;
    
};
}}
#endif