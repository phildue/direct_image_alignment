#include <execution>

#include "utils/utils.h"
#include "core/core.h"

namespace pd{namespace vision{

    template<typename Warp>
    LukasKanadeInverseCompositionalStacked<Warp>::LukasKanadeInverseCompositionalStacked ( const std::vector<std::shared_ptr<LukasKanadeInverseCompositional<Warp>>>& frames)
    : vslam::solver::Problem<SE3d::DoF>()
    , _frames(frames)
    { }

    template<typename Warp>
    void LukasKanadeInverseCompositionalStacked<Warp>::updateX(const Eigen::Matrix<double,Warp::nParameters,1>& dx)
    {
        std::for_each(_frames.begin(),_frames.end(),[&dx](auto f){f->updateX(dx);});
    }
    template<typename Warp>
    void LukasKanadeInverseCompositionalStacked<Warp>::setX(const Eigen::Matrix<double,Warp::nParameters,1>& x)
    {
        std::for_each(_frames.begin(),_frames.end(),[&x](auto f){f->setX(x);});
    }
    template<typename Warp>
    typename vslam::solver::NormalEquations<Warp::nParameters>::ConstShPtr LukasKanadeInverseCompositionalStacked<Warp>::computeNormalEquations() 
    {
        auto ne = std::make_shared<vslam::solver::NormalEquations<Warp::nParameters>>();
        std::for_each(std::execution::unseq,_frames.begin(),_frames.end(),[&](auto f){ne->combine(*f->computeNormalEquations());});
        return ne;
    }
   
}}