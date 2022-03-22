
#include "utils/utils.h"
#include "core/core.h"
#include <algorithm>
#include <execution>
namespace pd{namespace vision{

    template<typename Warp>
    LukasKanadeInverseCompositional<Warp>::LukasKanadeInverseCompositional (const Image& templ,const MatXi dTx, const MatXi dTy, const Image& image,std::shared_ptr<Warp> w0, std::shared_ptr<vslam::solver::Loss> l, double minGradient, vslam::solver::Scaler::ShPtr scaler)
    : vslam::solver::Problem<Warp::nParameters>()
    , _T(templ)
    , _I(image)
    , _w(w0)
    , _l(l)
    , _scaler(scaler)
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
                if( std::sqrt(dTx(v,u)*dTx(v,u)+dTy(v,u)*dTy(v,u)) >= _minGradient)
                {
                    _interestPoints.push_back({u,v,idx++});
                }
                    
            }
        }
        _J.conservativeResize(_interestPoints.size(),Eigen::NoChange);
         Eigen::MatrixXd steepestDescent = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
        std::for_each(std::execution::par_unseq,_interestPoints.begin(),_interestPoints.end(),[&](auto kp)
            {
                const Eigen::Matrix<double, 2,nParameters> Jw = _w->J(kp.u,kp.v);
                        
                _J.row(kp.idx) = dTx(kp.v, kp.u) * Jw.row(0) + dTy(kp.v,kp.u) * Jw.row(1);
                steepestDescent(kp.v,kp.u) = _J.row(kp.idx).norm();
            }
        );

        LOG_IMG("SteepestDescent") << steepestDescent;
    }
    template<typename Warp>
    LukasKanadeInverseCompositional<Warp>::LukasKanadeInverseCompositional (const Image& templ, const Image& image,std::shared_ptr<Warp> w0, std::shared_ptr<vslam::solver::Loss> l, double minGradient, vslam::solver::Scaler::ShPtr scaler)
    : LukasKanadeInverseCompositional<Warp> (templ, algorithm::gradX(templ), algorithm::gradY(templ), image, w0, l, minGradient, scaler){}



    template<typename Warp>
    void LukasKanadeInverseCompositional<Warp>::updateX(const Eigen::Matrix<double,Warp::nParameters,1>& dx)
    {
        _w->updateCompositional(-dx);
    }

    template<typename Warp>
    typename vslam::solver::NormalEquations<Warp::nParameters>::ConstShPtr LukasKanadeInverseCompositional<Warp>::computeNormalEquations() 
    {
        VecXd r = VecXd::Zero(_interestPoints.size());
        VecXd w = VecXd::Zero(_interestPoints.size());

        MatXi IWxp = MatXi::Zero(_I.rows(),_I.cols());
        Eigen::MatrixXd rImg = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
        Eigen::MatrixXd wImg = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
      
        std::for_each(std::execution::par_unseq,_interestPoints.begin(),_interestPoints.end(),[&](auto kp)
            {
                Eigen::Vector2d uvI = _w->apply(kp.u,kp.v);
                if (1 < uvI.x() && uvI.x() < _I.cols() -1  &&
                    1 < uvI.y() && uvI.y() < _I.rows()-1)
                {
                    // TODO just fill images and reshape at the end?
                    IWxp(kp.v,kp.u) =  algorithm::bilinearInterpolation(_I,uvI.x(),uvI.y());
                    r(kp.idx) = (double)IWxp(kp.v,kp.u) - (double)_T(kp.v,kp.u);
                    rImg(kp.v,kp.u) = r(kp.idx);
                    wImg(kp.v,kp.u) = 1.0;
                }else{
                    r(kp.idx) = std::numeric_limits<double>::quiet_NaN();
                    w(kp.idx) = std::numeric_limits<double>::quiet_NaN();

                }
            }
        );
       
        const Eigen::VectorXd rScaled = _scaler->scale(r); 

        std::for_each(std::execution::par_unseq,_interestPoints.begin(),_interestPoints.end(),[&](auto kp)
        {
            if (wImg(kp.v,kp.u) > 0.0)
            {
                w(kp.idx) = _l->computeWeight(rScaled(kp.idx));
                wImg(kp.v,kp.u) = w(kp.idx);
            }
        }
        );

        r = r.array().isNaN().select(0,r);
        w = w.array().isNaN().select(0,w);

       
        LOG_IMG("ImageWarped") << IWxp;
        LOG_IMG("Residual") << rImg;
        LOG_IMG("Weights") << wImg;

        auto ne = std::make_shared<vslam::solver::NormalEquations<Warp::nParameters>>();
        ne->A = _J.transpose() * w.asDiagonal() * _J;//todo: compute incrementally?
        ne->b = _J.transpose() * w.asDiagonal() * r;
        ne->chi2 = r.transpose() * w.asDiagonal() * r;
        ne->nConstraints = _interestPoints.size();
        return ne;
    }

}}

/*
       Image IWxp = Image::Zero(_I.rows(),_I.cols());
        std::vector<InterestPoint> interestPointsValid(_interestPoints.size());
        auto it = std::copy_if(std::execution::par_unseq,_interestPoints.begin(),_interestPoints.end(),interestPointsValid.begin(),
        [&](auto kp) {
                Eigen::Vector2d uvI = _w->apply(kp.u,kp.v);
                if (1 < uvI.x() && uvI.x() < _I.cols() - 1  &&
                    1 < uvI.y() && uvI.y() < _I.rows() - 1)
                {
                    IWxp(kp.v,kp.u) =  algorithm::bilinearInterpolation(_I,uvI.x(),uvI.y());
                    return true;
                }
                return false;
            }
        );
        interestPointsValid.resize(std::distance(interestPointsValid.begin(),it));

        MatXd R = IWxp.cast<double>() - _T.cast<double>();
        std::vector<double> r(interestPointsValid.size());
        std::transform(std::execution::unseq,interestPointsValid.begin(),interestPointsValid.end(),r.begin(),[&](auto kp){
            return R(kp.v,kp.u);
        });
       
        const Eigen::VectorXd rScaled = _scaler->scale(Eigen::Map<Eigen::VectorXd> (r.data(),r.size())); 
        
        auto ne = std::make_shared<vslam::solver::NormalEquations<Warp::nParameters>>();
        Eigen::MatrixXd W = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
        for(size_t i = 0; i < interestPointsValid.size(); i++)
        {
            const auto& kp = interestPointsValid[i];
            W(kp.v,kp.u) = _l->computeWeight(rScaled(i));
            ne->addConstraint(_J.row(kp.idx),R(kp.v,kp.u),W(kp.v,kp.u));
        }

        LOG_IMG("ImageWarped") << IWxp;
        LOG_IMG("Residual") << R;
        LOG_IMG("Weights") << W;

        return ne;


*/