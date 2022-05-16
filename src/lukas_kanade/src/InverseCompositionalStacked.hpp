#include <execution>
#include <vector>

#include "utils/utils.h"
#include "core/core.h"

namespace pd::vslam::lukas_kanade{

    template<typename Warp>
    InverseCompositionalStacked<Warp>::InverseCompositionalStacked ( const std::vector<std::shared_ptr<InverseCompositional<Warp>>>& frames)
    : least_squares::Problem(SE3d::DoF)
    , _frames(frames)
    { }

    template<typename Warp>
    void InverseCompositionalStacked<Warp>::updateX(const Eigen::VectorXd& dx)
    {
        std::for_each(_frames.begin(),_frames.end(),[&dx](auto f){f->updateX(dx);});
    }
    template<typename Warp>
    void InverseCompositionalStacked<Warp>::setX(const Eigen::VectorXd& x)
    {
        std::for_each(_frames.begin(),_frames.end(),[&x](auto f){f->setX(x);});
    }
    template<typename Warp>
    least_squares::NormalEquations::ConstShPtr InverseCompositionalStacked<Warp>::computeNormalEquations() 
    {
        std::vector<least_squares::NormalEquations::ConstShPtr> nes(_frames.size());
        std::transform(_frames.begin(),_frames.end(),nes.begin(),[&](auto f){return f->computeNormalEquations();});
        auto ne = std::make_shared<least_squares::NormalEquations>(Warp::nParameters);
        std::for_each(nes.begin(),nes.end(),[&](auto n){ne->combine(*n);});
        return ne;
    }
   
}