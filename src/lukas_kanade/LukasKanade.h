#ifndef LUKAS_KANADE_H__
#define LUKAS_KANADE_H__
#include "core/types.h"
namespace pd{namespace vision{
class LukasKanade{
public:
    LukasKanade (const Image& templ, const Image& image);
   
    //
    // r = T(x) - I(W(x,p))
    //
    bool computeResidual(const Eigen::VectorXd& x, Eigen::VectorXd& r, Eigen::VectorXd& w) const;
   
    //
    // J = Ixy*dW/dp
    //
    bool computeJacobian(const Eigen::VectorXd& x, Eigen::MatrixXd& j) const;

    bool updateX(const Eigen::VectorXd& dx, Eigen::VectorXd& x) const;
protected:
    const Image _T;
    const Image _Iref;
    Eigen::MatrixXi _dIx;
    Eigen::MatrixXi _dIy;
    LevenbergMarquardt _solver;
    
};
}}
#endif