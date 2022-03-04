
#include "utils/utils.h"
#include "core/core.h"
#include <algorithm>
#include <execution>
namespace pd{namespace vision{

    template<typename Warp>
    LukasKanadeInverseCompositional<Warp>::LukasKanadeInverseCompositional (const Image& templ, const Image& image,std::shared_ptr<Warp> w0, std::shared_ptr<Loss> l, double minGradient)
    : _T(templ)
    , _Iref(image)
    , _w(w0)
    , _J(Eigen::MatrixXd::Zero(_Iref.rows()*_Iref.cols(),Warp::nParameters))
    , _l(l)
    , _dTx(algorithm::gradX(templ))
    , _dTy(algorithm::gradY(templ))
    , _dTxy(MatXd::Zero(_Iref.rows(),_Iref.cols()))
    , _minGradient(minGradient)
    {
        Eigen::MatrixXd steepestDescent = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
        //TODO this could come from some external feature selector
        //TODO move dTx, dTy computation outside
        _interestPoints.reserve(_T.rows() *_T.cols());
        uint32_t idx=0;
        for (uint32_t v = 0; v < _T.rows(); v++)
        {
            for (uint32_t u = 0; u < _T.cols(); u++)
            {
                _dTxy(v,u) = std::sqrt(_dTx(v,u)*_dTx(v,u)+_dTy(v,u)*_dTy(v,u));
                if( _dTxy(v,u) >= _minGradient)
                {
                    _interestPoints.push_back({u,v,idx++});
                }
                    
            }
        }
        _J.conservativeResize(_interestPoints.size(), Eigen::NoChange);
        std::for_each(std::execution::par_unseq,_interestPoints.begin(),_interestPoints.end(),[&](auto kp)
            {
                const Eigen::Matrix<double, 2,nParameters> Jwarp = _w->J(kp.u,kp.v);
                        
                auto j = _dTx(kp.v, kp.u) * Jwarp.row(0) + _dTy(kp.v,kp.u) * Jwarp.row(1);
                steepestDescent(kp.v,kp.u) = j.norm();
                _J.row(kp.idx) = j;
            }
        );

        LOG_IMG("DTX") << _dTx;
        LOG_IMG("DTY") << _dTy;
        LOG_IMG("SteepestDescent") << steepestDescent;
    }

    template<typename Warp>
    void LukasKanadeInverseCompositional<Warp>::computeResidual(Eigen::VectorXd& r, Eigen::VectorXd& w)
    {

        Eigen::MatrixXd rImg = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
        Eigen::MatrixXd wImg = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
        Image IWxp = Image::Zero(_Iref.rows(),_Iref.cols());
        std::vector<double> validRs(_interestPoints.size());
        
        r.conservativeResize(_interestPoints.size());
        w.conservativeResize(_interestPoints.size());

        w.setZero();
        r.setZero();

        std::for_each(std::execution::par_unseq,_interestPoints.begin(),_interestPoints.end(),[&](auto kp)
            {
                Eigen::Vector2d uvWarped = _w->apply(kp.u,kp.v);
                if (1 < uvWarped.x() && uvWarped.x() < _Iref.cols() -1  &&
                1 < uvWarped.y() && uvWarped.y() < _Iref.rows()-1)
                {
                    // TODO just fill images and reshape at the end?
                    IWxp(kp.v,kp.u) =  algorithm::bilinearInterpolation(_Iref,uvWarped.x(),uvWarped.y());
                    r(kp.idx) = IWxp(kp.v,kp.u) - _T(kp.v,kp.u);
                    rImg(kp.v,kp.u) = r(kp.idx);
                    wImg(kp.v,kp.u) = 1.0;
                    validRs[kp.idx] = r(kp.idx);
                }else{
                    validRs[kp.idx] = std::numeric_limits<double>::quiet_NaN();
                }
            }
        );
        validRs.erase(std::remove_if(validRs.begin(),validRs.end(),[](auto vR){ return !std::isfinite(vR); }),validRs.end());

        const Eigen::Map<Eigen::VectorXd> rValid(validRs.data(),validRs.size());
        double median = algorithm::median(rValid);
        const auto stddev = (rValid.array() - median).array().abs().sum()/(rValid.rows() - 1);
        const Eigen::VectorXd rScaled = (r.array() - median)/stddev;

        std::for_each(std::execution::par_unseq,_interestPoints.begin(),_interestPoints.end(),[&](auto kp)
        {
            if (wImg(kp.v,kp.u) > 0.0)
            {
                w(kp.idx) = _l->computeWeight(rScaled(kp.idx));
                wImg(kp.v,kp.u) = w(kp.idx);
            }
        }
        );

       
        LOG_IMG("ImageWarped") << IWxp;
        LOG_IMG("Residual") << rImg;
        LOG_IMG("Weights") << wImg;
    }

    //
    // J = Ixy*dW/dp
    //
    template<typename Warp>
    bool LukasKanadeInverseCompositional<Warp>::computeJacobian(Eigen::Matrix<double, -1,Warp::nParameters>& j)
    {
        j = _J;
        return true;
    }

    

    template<typename Warp>
    bool LukasKanadeInverseCompositional<Warp>::updateX(const Eigen::Matrix<double,Warp::nParameters,1>& dx)
    {
        _w->updateCompositional(-dx);
        return true;
    }

    template<typename Warp>
    void LukasKanadeInverseCompositional<Warp>::extendLeft(Eigen::MatrixXd& UNUSED(H))
    {}
    template<typename Warp>
    void LukasKanadeInverseCompositional<Warp>::extendRight(Eigen::VectorXd& UNUSED(g))
    {}

}}