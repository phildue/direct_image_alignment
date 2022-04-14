#ifndef VSLAM_LUKAS_KANADE_INVERSE_COMPOSITIONAL_STACKED_H__
#define VSLAM_LUKAS_KANADE_INVERSE_COMPOSITIONAL_STACKED_H__

#include <memory>
#include <vector>

#include "core/core.h"
#include "solver/solver.h"

#include "LukasKanadeInverseCompositional.h"

namespace pd{namespace vision{

template<typename Warp>
class LukasKanadeInverseCompositionalStacked : public vslam::solver::Problem<Warp::nParameters>{
    
public:
    LukasKanadeInverseCompositionalStacked (const std::vector<std::shared_ptr<LukasKanadeInverseCompositional<Warp>>>& frames);
    std::shared_ptr<const Warp> warp() { return _frames[0]->warp();}

    void updateX(const Eigen::Matrix<double,Warp::nParameters,1>& dx) override;

    Eigen::Matrix<double,Warp::nParameters,1> x() const override{return _frames[0]->x();}
    void setX(const Eigen::Matrix<double,Warp::nParameters,1>& x) override;

    typename vslam::solver::NormalEquations<Warp::nParameters>::ConstShPtr computeNormalEquations() override;

protected:
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
