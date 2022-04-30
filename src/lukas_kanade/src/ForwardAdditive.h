#ifndef VSLAM_LUKAS_KANADE_H__
#define VSLAM_LUKAS_KANADE_H__
#include "core/core.h"
#include "least_squares/least_squares.h"
#include <memory>
namespace pd::vslam::lukas_kanade{

template<typename Warp>
class ForwardAdditive : public least_squares::Problem<Warp::nParameters>{
public:
    inline constexpr static int nParameters = Warp::nParameters;
    ForwardAdditive (const Image& templ, const MatXd& dX, const MatXd& dY, const Image& image,
     std::shared_ptr<Warp> w0,
     least_squares::Loss::ShPtr = std::make_shared<least_squares::QuadraticLoss>(),
     double minGradient = 0,
     std::shared_ptr<const least_squares::Prior<Warp::nParameters>> prior = nullptr);
    const std::shared_ptr<const Warp> warp();

    void updateX(const Eigen::Matrix<double,Warp::nParameters,1>& dx) override;
    void setX(const Eigen::Matrix<double,Warp::nParameters,1>& x) override {_w->setX(x);}

    Eigen::Matrix<double,Warp::nParameters,1> x() const override {return _w->x();}
    typename least_squares::NormalEquations<Warp::nParameters>::ConstShPtr computeNormalEquations() override;


protected:

    const Image _T;
    const Image _Iref;
    Eigen::MatrixXd _dIdx;
    Eigen::MatrixXd _dIdy;
    const std::shared_ptr<Warp> _w;
    const std::shared_ptr<least_squares::Loss> _loss;
    const double _minGradient;
    const std::shared_ptr<const least_squares::Prior<Warp::nParameters>> _prior;

    std::vector<Eigen::Vector2i> _interestPoints;
};

}
#include "ForwardAdditive.hpp"
#include "Warp.h"
namespace pd::vslam::lukas_kanade{

typedef ForwardAdditive<WarpAffine> ForwardAdditiveAffine;
typedef ForwardAdditive<WarpOpticalFlow> ForwardAdditiveOpticalFlow;
typedef ForwardAdditive<WarpSE3> ForwardAdditiveSE3;
}
#endif
