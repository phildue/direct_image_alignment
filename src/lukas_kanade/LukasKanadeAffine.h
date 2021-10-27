#ifndef VSLAM_LUKAS_KANADE_AFFINE_H__
#define VSLAM_LUKAS_KANADE_AFFINE_H__
#include <memory>
#include "LukasKanade.h"
#include "core/types.h"
#include "solver/solver.h"
namespace pd{namespace vision{
class LukasKanadeAffine : public LukasKanade<6>{
public:
    LukasKanadeAffine (const Image& templ, const Image& image, int maxIterations, double minStepSize = 1e-3, double minGradient = 1e-3);

protected:

    bool updateX(const Eigen::Vector6d& dx, Eigen::Vector6d& x) const;

    Eigen::MatrixXi warp(const Eigen::Matrix<double,6,1>& x, const Eigen::MatrixXi& img) const override;
    Image warp(const Eigen::Matrix<double,6,1>& x, const Image& img) const override;

    Eigen::Matrix<double,2,6> jacobianWarp(int v, int u) const override;

    const Image _T;
    const Image _Iref;
    Eigen::MatrixXi _dIx;
    Eigen::MatrixXi _dIy;
    std::shared_ptr<const Solver<6>> _solver;
    
};
}}
#endif