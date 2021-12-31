#ifndef VSLAM_LUKAS_KANADE_INVERSE_COMPOSITIONAL_H__
#define VSLAM_LUKAS_KANADE_INVERSE_COMPOSITIONAL_H__
#include "core/types.h"
#include "solver/solver.h"
#include <memory>
namespace pd{namespace vision{

template<typename Warp>
class LukasKanadeInverseCompositional{
public:
    inline constexpr static int nParameters = Warp::nParameters;
    LukasKanadeInverseCompositional (const Image& templ, const Image& image, std::shared_ptr<Warp> w0);
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
    const std::shared_ptr<Warp> _w;
    Image _IWxp;
    Eigen::Matrix<double,-1,Warp::nParameters> _J;
    Eigen::MatrixXi _dIx,_dIy;
};

}}
#include "LukasKanadeInverseCompositional.hpp"
#endif
