#ifndef VSLAM_LUKAS_KANADE_H__
#define VSLAM_LUKAS_KANADE_H__
#include "core/core.h"
#include "solver/solver.h"
#include <memory>
namespace pd{namespace vision{

template<typename Warp>
class LukasKanade : public vslam::solver::Problem<Warp::nParameters>{
public:
    inline constexpr static int nParameters = Warp::nParameters;
    LukasKanade (const Image& templ, const MatXd& dX, const MatXd& dY, const Image& image,
     std::shared_ptr<Warp> w0,
     vslam::solver::Loss::ShPtr = std::make_shared<vslam::solver::QuadraticLoss>(),
     double minGradient = 0,
     std::shared_ptr<const vslam::solver::Prior<Warp::nParameters>> prior = nullptr);
    const std::shared_ptr<const Warp> warp();

    void updateX(const Eigen::Matrix<double,Warp::nParameters,1>& dx) override;
    void setX(const Eigen::Matrix<double,Warp::nParameters,1>& x) override {_w->setX(x);}

    Eigen::Matrix<double,Warp::nParameters,1> x() const override {return _w->x();}
    Eigen::Matrix<double,Warp::nParameters,Warp::nParameters> cov() const override {return _covariance;}
    typename vslam::solver::NormalEquations<Warp::nParameters>::ConstShPtr computeNormalEquations() override;


protected:

    const Image _T;
    const Image _Iref;
    Eigen::MatrixXd _dIdx;
    Eigen::MatrixXd _dIdy;
    const std::shared_ptr<Warp> _w;
    const std::shared_ptr<vslam::solver::Loss> _loss;
    const double _minGradient;
    const std::shared_ptr<const vslam::solver::Prior<Warp::nParameters>> _prior;

    std::vector<Eigen::Vector2i> _interestPoints;
    Eigen::Matrix<double,Warp::nParameters,Warp::nParameters> _covariance;
};

}}
#include "LukasKanade.hpp"
#include "Warp.h"
namespace pd{namespace vision{

typedef LukasKanade<WarpAffine> LukasKanadeAffine;
typedef LukasKanade<WarpOpticalFlow> LukasKanadeOpticalFlow;
typedef LukasKanade<WarpSE3> LukasKanadeSE3;
}}
#endif
