
#include "utils/utils.h"
#include "core/core.h"
#include <algorithm>
#include <execution>
namespace pd{namespace vision{

    template<typename Warp>
    LukasKanadeInverseCompositional<Warp>::LukasKanadeInverseCompositional (const Image& templ,const MatXi dTx, const MatXi dTy, const Image& image,std::shared_ptr<Warp> w0, std::shared_ptr<vslam::solver::Loss> l, double minGradient, vslam::solver::Scaler::ShPtr scaler)
    : _T(templ)
    , _I(image)
    , _w(w0)
    , _l(l)
    , _scaler(scaler)
    , _dTx(dTx)
    , _dTy(dTy)
    , _dTxy(MatXd::Zero(_I.rows(),_I.cols()))
    , _minGradient(minGradient)
    {
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
    }
    template<typename Warp>
    LukasKanadeInverseCompositional<Warp>::LukasKanadeInverseCompositional (const Image& templ, const Image& image,std::shared_ptr<Warp> w0, std::shared_ptr<vslam::solver::Loss> l, double minGradient, vslam::solver::Scaler::ShPtr scaler)
    : LukasKanadeInverseCompositional<Warp> (templ, algorithm::gradX(templ), algorithm::gradY(templ), image, w0, l, minGradient, scaler){}

    template<typename Warp>
    void LukasKanadeInverseCompositional<Warp>::computeResidual(Eigen::VectorXd& r, Eigen::VectorXd& w)
    {
        computeResidual(r,w,0);
    }
    template<typename Warp>
    void LukasKanadeInverseCompositional<Warp>::computeResidual(Eigen::VectorXd& r, Eigen::VectorXd& w, size_t offset)
    {
        Image IWxp = Image::Zero(_I.rows(),_I.cols());
        Eigen::MatrixXd rImg = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
        Eigen::MatrixXd wImg = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
      
        std::for_each(std::execution::par_unseq,_interestPoints.begin(),_interestPoints.end(),[&](auto kp)
            {
                Eigen::Vector2d uvWarped = _w->apply(kp.u,kp.v);
                if (1 < uvWarped.x() && uvWarped.x() < _I.cols() -1  &&
                1 < uvWarped.y() && uvWarped.y() < _I.rows()-1)
                {
                    // TODO just fill images and reshape at the end?
                    IWxp(kp.v,kp.u) =  algorithm::bilinearInterpolation(_I,uvWarped.x(),uvWarped.y());
                    r(offset + kp.idx) = IWxp(kp.v,kp.u) - _T(kp.v,kp.u);
                    rImg(kp.v,kp.u) = r(kp.idx);
                    wImg(kp.v,kp.u) = 1.0;
                }else{
                    r(offset + kp.idx) = std::numeric_limits<double>::quiet_NaN();
                }
            }
        );
       
        const Eigen::VectorXd rScaled = _scaler->scale(r); 

        std::for_each(std::execution::par_unseq,_interestPoints.begin(),_interestPoints.end(),[&](auto kp)
        {
            if (wImg(kp.v,kp.u) > 0.0)
            {
                w(offset + kp.idx) = _l->computeWeight(rScaled(offset + kp.idx));
                wImg(kp.v,kp.u) = w(kp.idx);
            }
        }
        );
        for(size_t i = 0; i < _interestPoints.size(); i++)
        {
            if(!std::isfinite(r(offset + i)))
            {
                r(offset + i) = 0.0;
                w(offset + i) = 0.0;
            }
        }
       
        LOG_IMG("ImageWarped") << IWxp;
        LOG_IMG("Residual") << rImg;
        LOG_IMG("Weights") << wImg;

    }

    template<typename Warp>
    double LukasKanadeInverseCompositional<Warp>::computeResidual(size_t idx)
    {
        const auto& kp = _interestPoints[idx];
        Eigen::Vector2d uvWarped = _w->apply(kp.u,kp.v);
        if (1 < uvWarped.x() && uvWarped.x() < _I.cols() -1  &&
        1 < uvWarped.y() && uvWarped.y() < _I.rows()-1)
        {
            return algorithm::bilinearInterpolation(_I,uvWarped.x(),uvWarped.y()) - _T(kp.v,kp.u);
            
        }else{
            return std::numeric_limits<double>::quiet_NaN();
        }
    }

    //
    // J = Ixy*dW/dp
    //
    template<typename Warp>
    bool LukasKanadeInverseCompositional<Warp>::computeJacobian(Eigen::Matrix<double, -1,Warp::nParameters>& j)
    {
        computeJacobian(j,0U);
        return true;
    }

    template<typename Warp>
    bool LukasKanadeInverseCompositional<Warp>::computeJacobian(Eigen::Matrix<double, -1,Warp::nParameters>& J, size_t offset)
    {
        Eigen::MatrixXd steepestDescent = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
        std::for_each(std::execution::par_unseq,_interestPoints.begin(),_interestPoints.end(),[&](auto kp)
            {
                const Eigen::Matrix<double, 2,nParameters> Jwarp = _w->J(kp.u,kp.v);
                        
                auto j = _dTx(kp.v, kp.u) * Jwarp.row(0) + _dTy(kp.v,kp.u) * Jwarp.row(1);
                steepestDescent(kp.v,kp.u) = j.norm();
                J.row(offset + kp.idx) = j;
            }
        );

        LOG_IMG("SteepestDescent") << steepestDescent;
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