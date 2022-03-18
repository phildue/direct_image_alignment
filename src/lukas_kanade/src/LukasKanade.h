#ifndef VSLAM_LUKAS_KANADE_H__
#define VSLAM_LUKAS_KANADE_H__
#include "core/core.h"
#include "solver/solver.h"
#include <memory>
namespace pd{namespace vision{

template<typename Warp>
class LukasKanade{
public:
    inline constexpr static int nParameters = Warp::nParameters;
    LukasKanade (const Image& templ, const Image& image, std::shared_ptr<Warp> w0, std::shared_ptr<vslam::solver::Loss> l = std::make_shared<vslam::solver::QuadraticLoss>());
    const std::shared_ptr<const Warp> warp();
    bool newJacobian() const {return true;}

      //
    // r = T(x) - I(W(x,p))
    //
    void computeResidual(Eigen::VectorXd& r, Eigen::VectorXd& w);

   
    //
    // J = Ixy*dW/dp
    //
    bool computeJacobian(Eigen::Matrix<double,-1,Warp::nParameters>& j) ;

    bool updateX(const Eigen::Matrix<double,Warp::nParameters,1>& dx);

    Eigen::Matrix<double,Warp::nParameters,1> x() const {return _w->x();}
    void extendLeft(Eigen::MatrixXd& H);
    void extendRight(Eigen::VectorXd& g);

protected:

    const Image _T;
    const Image _Iref;
    Eigen::MatrixXi _dIx;
    Eigen::MatrixXi _dIy;
    const std::shared_ptr<Warp> _w;
    const std::shared_ptr<vslam::solver::Loss> _l;
    const std::shared_ptr<vslam::solver::Scaler> _scaler;

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
