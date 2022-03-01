#ifndef VSLAM_LUKAS_KANADE_INVERSE_COMPOSITIONAL_STACKED_H__
#define VSLAM_LUKAS_KANADE_INVERSE_COMPOSITIONAL_STACKED_H__

#include <memory>
#include <vector>

#include "core/core.h"
#include "solver/solver.h"

#include "LukasKanadeInverseCompositional.h"

namespace pd{namespace vision{

template<typename Warp>
class LukasKanadeInverseCompositionalStacked{
    
public:
    inline constexpr static int nParameters = Warp::nParameters;
    LukasKanadeInverseCompositionalStacked (const std::vector<Image>& templ, const Image& image,const std::vector<std::shared_ptr<Warp>>& w0, std::shared_ptr<Loss> = std::make_shared<QuadraticLoss>(), double minGradient = 0);
    const std::shared_ptr<const Warp> warp();

      //
    // r = T(x) - I(W(x,p))
    //
    void computeResidual(Eigen::VectorXd& r, Eigen::VectorXd& w);
   
    //
    // J = Ixy*dW/dp
    //
    bool computeJacobian(Eigen::Matrix<double,-1,Warp::nParameters>& j) ;

    bool updateX(const Eigen::Matrix<double,Warp::nParameters,1>& dx);

    void extendLeft(Eigen::MatrixXd& H);
    void extendRight(Eigen::VectorXd& g);

    Eigen::Matrix<double,Warp::nParameters,1> x() const {return _frames[0]->x();}

protected:
    Eigen::MatrixXd _J;
    std::vector<std::shared_ptr<LukasKanadeInverseCompositional<Warp>>> _frames;

};

}}
#include "LukasKanadeInverseCompositionalStacked.hpp"
#include "Warp.h"
namespace pd{namespace vision{

typedef LukasKanadeInverseCompositionalStacked<WarpAffine> LukasKanadeInverseCompositionalStackedlAffine;
typedef LukasKanadeInverseCompositionalStacked<WarpOpticalFlow> LukasKanadeInverseCompositionalStackedOpticalFlow;
typedef LukasKanadeInverseCompositionalStacked<WarpSE3> LukasKanadeInverseCompositionalStackedSE3;
}}
#endif
