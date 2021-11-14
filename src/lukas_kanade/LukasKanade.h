#ifndef VSLAM_LUKAS_KANADE_H__
#define VSLAM_LUKAS_KANADE_H__
#include "core/types.h"
#include "solver/solver.h"
#include <memory>
namespace pd{namespace vision{

template<typename Warp>
class LukasKanade{
public:
    inline constexpr static int nParameters = Warp::nParameters;
    LukasKanade (const Image& templ, const Image& image, std::shared_ptr<Warp> w0);
    const std::shared_ptr<const Warp> warp();

      //
    // r = T(x) - I(W(x,p))
    //
    bool computeResidual(Eigen::VectorXd& r);
   
    //
    // J = Ixy*dW/dp
    //
    bool computeJacobian(Eigen::Matrix<double,-1,Warp::nParameters>& j) ;

    bool updateX(const Eigen::Matrix<double,Warp::nParameters,1>& dx);

    Eigen::Matrix<double,Warp::nParameters,1> x() const {return _w->x();}

protected:

    const Image _T;
    const Image _Iref;
    Eigen::MatrixXi _dIx;
    Eigen::MatrixXi _dIy;
    const std::shared_ptr<Warp> _w;

};
}}
#include "LukasKanade.hpp"
#include "WarpAffine.h"
#include "WarpOpticalFlow.h"
#include "WarpSE3.h"
namespace pd{namespace vision{

typedef LukasKanade<WarpAffine> LukasKanadeAffine;
typedef LukasKanade<WarpOpticalFlow> LukasKanadeOpticalFlow;
typedef LukasKanade<WarpSE3> LukasKanadeSE3;
}}
#endif
