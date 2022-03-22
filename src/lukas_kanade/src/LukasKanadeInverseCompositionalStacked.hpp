
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
    typename vslam::solver::NormalEquations<Warp::nParameters>::ConstShPtr LukasKanadeInverseCompositionalStacked<Warp>::computeNormalEquations() 
    {
        auto ne = std::make_shared<vslam::solver::NormalEquations<Warp::nParameters>>();
        ne->A.setZero();
        ne->b.setZero();
        ne->chi2 = 0;
        ne->nConstraints = 0;
        for(size_t i = 0; i < _frames.size(); i++)
        {
            auto ne_i = _frames[i]->computeNormalEquations();
            ne->A.noalias() +=  ne_i->A;
            ne->b.noalias() +=  ne_i->b;
            ne->chi2 +=  ne_i->chi2;
            ne->nConstraints +=  ne_i->nConstraints;
        
        }
        return ne;

    }
   
}}