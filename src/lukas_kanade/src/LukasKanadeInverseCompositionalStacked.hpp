
#include "utils/utils.h"
#include "core/core.h"

namespace pd{namespace vision{

    template<typename Warp>
    LukasKanadeInverseCompositionalStacked<Warp>::LukasKanadeInverseCompositionalStacked (const std::vector<Image>& templ,const std::vector<MatXi>& dTx, const std::vector<MatXi>& dTy, const Image& image,const std::vector<std::shared_ptr<Warp>>& w0, std::shared_ptr<vslam::solver::Loss> l, double minGradient,vslam::solver::Scaler::ShPtr scaler)
    {
        if ( templ.size() != w0.size() )
        {
            throw pd::Exception(" Each template needs a warp ");
        }
        _frames.resize(templ.size());
        _nConstraints = 0;
        for ( size_t i = 0; i < _frames.size(); i++)
        {
            _frames[i] = std::make_shared<LukasKanadeInverseCompositional<Warp>>(templ[i], dTx[i], dTy[i], image, w0[i], l, minGradient, scaler);
            _nConstraints += _frames[i]->nConstraints();
        }
    }

    template<typename Warp>
    void LukasKanadeInverseCompositionalStacked<Warp>::computeResidual(Eigen::VectorXd& r, Eigen::VectorXd& w)
    {
        size_t idx = 0U;
        for ( size_t i = 0; i < _frames.size(); i++ )
        {
            _frames[i]->computeResidual(r,w,idx);
            idx += _frames[i]->nConstraints();
        }
    }

    //
    // J = Ixy*dW/dp
    //
    template<typename Warp>
    bool LukasKanadeInverseCompositionalStacked<Warp>::computeJacobian(Eigen::Matrix<double, -1,Warp::nParameters>& j)
    {
        size_t idx = 0U;
        for ( size_t i = 0; i < _frames.size(); i++ )
        {
            _frames[i]->computeJacobian(j,idx);
            idx += _frames[i]->nConstraints();
        }
        return true;
    }

    template<typename Warp>
    bool LukasKanadeInverseCompositionalStacked<Warp>::updateX(const Eigen::Matrix<double,Warp::nParameters,1>& dx)
    {
        for ( auto& f : _frames )
        {
            f->updateX(dx);
        }
        return true;
    }

    template<typename Warp>
    void LukasKanadeInverseCompositionalStacked<Warp>::extendLeft(Eigen::MatrixXd& UNUSED(H))
    {}
    template<typename Warp>
    void LukasKanadeInverseCompositionalStacked<Warp>::extendRight(Eigen::VectorXd& UNUSED(g))
    {}

}}