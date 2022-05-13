#ifndef VSLAM_LUKAS_KANADE_INVERSE_COMPOSITIONAL_STACKED_H__
#define VSLAM_LUKAS_KANADE_INVERSE_COMPOSITIONAL_STACKED_H__

#include <memory>
#include <vector>

#include "core/core.h"
#include "least_squares/least_squares.h"

#include "InverseCompositional.h"

namespace pd::vslam::lukas_kanade{

template<typename Warp>
class InverseCompositionalStacked : public least_squares::Problem<Warp::nParameters>{
    
public:
    InverseCompositionalStacked (const std::vector<std::shared_ptr<InverseCompositional<Warp>>>& frames);
    std::shared_ptr<const Warp> warp() { return _frames[0]->warp();}

    void updateX(const Eigen::Matrix<double,Warp::nParameters,1>& dx) override;

    Eigen::Matrix<double,Warp::nParameters,1> x() const override{return _frames[0]->x();}
    void setX(const Eigen::Matrix<double,Warp::nParameters,1>& x) override;

    least_squares::NormalEquations::ConstShPtr computeNormalEquations() override;

protected:
    std::vector<std::shared_ptr<InverseCompositional<Warp>>> _frames;

};

}
#include "InverseCompositionalStacked.hpp"
#include "Warp.h"
namespace pd::vslam::lukas_kanade{

typedef InverseCompositionalStacked<WarpAffine> InverseCompositionalStackedAffine;
typedef InverseCompositionalStacked<WarpOpticalFlow> InverseCompositionalStackedOpticalFlow;
typedef InverseCompositionalStacked<WarpSE3> InverseCompositionalStackedStackedSE3;
}
#endif
