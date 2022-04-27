#ifndef VSLAM_LUKAS_KANADE_INVERSE_COMPOSITIONAL_H__
#define VSLAM_LUKAS_KANADE_INVERSE_COMPOSITIONAL_H__
#include "core/core.h"
#include "least_squares/least_squares.h"
#include <memory>
namespace pd::vslam::lukas_kanade{

/*
Compute parameters p of the warp W by incrementally warping the image I to the template T.
For inverse compositional we switch the role of I and T:

chi2 = rho(|T(W(x,p+h)) - I(W(x,p))|^2)
with rho being a robust loss function reducing the influence of outliers
and h the parameter update.

This results in normal equations:

(J^T * W * J)h = -J*W*r           (1)

where Ji =  [dTix * JiWx, dTiy * JiWy]
with i corresponding to the ith pixel / feature / row in J
dTi* being the image derivative in x/y direction of the ith pixel / feature
JiW* being the partial derivatives of the warping function to its parameters.

Given Jw is independent of h, J can be precomputed as its not changing during a parameter update.

After solving (1) we obtain the parameter update that would warp the template to the image!
Hence, we update p by applying the *inverse compositional*: W_new = W(-h,W(p)).
*/

template<typename Warp>
class InverseCompositional : public least_squares::Problem<Warp::nParameters>{
public:
    inline constexpr static int nParameters = Warp::nParameters;
    InverseCompositional (const Image& templ, const MatXd& dX, const MatXd& dY, const Image& image,
    std::shared_ptr<Warp> w0,
    least_squares::Loss::ShPtr = std::make_shared<least_squares::QuadraticLoss>(),
    double minGradient = 0,
    std::shared_ptr<const least_squares::Prior<Warp::nParameters>> prior = nullptr);
    
    InverseCompositional (const Image& templ, const MatXd& dX, const MatXd& dY, const Image& image,
    std::shared_ptr<Warp> w0,
    const std::vector<Eigen::Vector2i>& interestPoints,
    least_squares::Loss::ShPtr = std::make_shared<least_squares::QuadraticLoss>(),
    std::shared_ptr<const least_squares::Prior<Warp::nParameters>> prior = nullptr);

    InverseCompositional (const Image& templ, const Image& image, std::shared_ptr<Warp> w0, 
    least_squares::Loss::ShPtr = std::make_shared<least_squares::QuadraticLoss>(), double minGradient = 0, std::shared_ptr<const least_squares::Prior<Warp::nParameters>> prior = nullptr);
    std::shared_ptr<const Warp> warp() { return _w;}

    void updateX(const Eigen::Matrix<double,Warp::nParameters,1>& dx) override;
    void setX(const Eigen::Matrix<double,Warp::nParameters,1>& x) override {_w->setX(x);}

    Eigen::Matrix<double,Warp::nParameters,1> x() const override {return _w->x();}

    typename least_squares::NormalEquations<Warp::nParameters>::ConstShPtr computeNormalEquations() override;

protected:
    const Image _T;
    const Image _I;
    const std::shared_ptr<Warp> _w;
    const std::shared_ptr<least_squares::Loss> _loss;
    const std::shared_ptr<const least_squares::Prior<Warp::nParameters>> _prior;
    Eigen::Matrix<double,-1,Warp::nParameters> _J;
    struct IndexedKeyPoint{
        size_t idx;
        Eigen::Vector2i pos;
    };
    std::vector<IndexedKeyPoint> _interestPoints;

};

}
#include "InverseCompositional.hpp"
#include "Warp.h"
namespace pd::vslam::lukas_kanade{

typedef InverseCompositional<WarpAffine> InverseCompositionalAffine;
typedef InverseCompositional<WarpOpticalFlow> InverseCompositionalOpticalFlow;
typedef InverseCompositional<WarpSE3>InverseCompositionalSE3;
}
#endif
