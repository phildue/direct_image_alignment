#ifndef VSLAM_LUKAS_KANADE_INVERSE_COMPOSITIONAL_H__
#define VSLAM_LUKAS_KANADE_INVERSE_COMPOSITIONAL_H__
#include "core/core.h"
#include "solver/solver.h"
#include <memory>
namespace pd{namespace vision{



template<typename Warp>
class LukasKanadeInverseCompositional{
public:
    inline constexpr static int nParameters = Warp::nParameters;
    LukasKanadeInverseCompositional (const Image& templ, const MatXi dX, const MatXi dY, const Image& image, std::shared_ptr<Warp> w0, vslam::solver::Loss::ShPtr = std::make_shared<vslam::solver::QuadraticLoss>(), double minGradient = 0, vslam::solver::Scaler::ShPtr scaler = std::make_shared<vslam::solver::Scaler>());

    LukasKanadeInverseCompositional (const Image& templ, const Image& image, std::shared_ptr<Warp> w0, vslam::solver::Loss::ShPtr = std::make_shared<vslam::solver::QuadraticLoss>(), double minGradient = 0, vslam::solver::Scaler::ShPtr scaler = std::make_shared<vslam::solver::Scaler>());
    const std::shared_ptr<const Warp> warp();
    size_t nConstraints() const { return _interestPoints.size();}
    bool newJacobian() const {return false;}
      //
    // r = T(x) - I(W(x,p))
    //
    void computeResidual(Eigen::VectorXd& r, Eigen::VectorXd& w);
    void computeResidual(Eigen::VectorXd& r, Eigen::VectorXd& w, size_t offset);
    double computeResidual(size_t idx);
    //
    // J = Ixy*dW/dp
    //
    bool computeJacobian(Eigen::Matrix<double,-1,Warp::nParameters>& j) ;
    bool computeJacobian(Eigen::Matrix<double,-1,Warp::nParameters>& J, size_t offset) ;

    bool updateX(const Eigen::Matrix<double,Warp::nParameters,1>& dx);

    void extendLeft(Eigen::MatrixXd& H);
    void extendRight(Eigen::VectorXd& g);

    Eigen::Matrix<double,Warp::nParameters,1> x() const {return _w->x();}

protected:
    struct InterestPoint{
            uint32_t u,v,idx;
        };
    const Image _T;
    const Image _I;
    const std::shared_ptr<Warp> _w;
    const std::shared_ptr<vslam::solver::Loss> _l;
    const std::shared_ptr<vslam::solver::Scaler> _scaler;

    const MatXi _dTx,_dTy;
    MatXd _dTxy;
    const double _minGradient;
    std::vector<InterestPoint> _interestPoints;

};

}}
#include "LukasKanadeInverseCompositional.hpp"
#include "Warp.h"
namespace pd{namespace vision{

typedef LukasKanadeInverseCompositional<WarpAffine> LukasKanadeInverseCompositionalAffine;
typedef LukasKanadeInverseCompositional<WarpOpticalFlow> LukasKanadeInverseCompositionalOpticalFlow;
typedef LukasKanadeInverseCompositional<WarpSE3> LukasKanadeInverseCompositionalSE3;
}}
#endif
