
#include "utils/utils.h"
#include "core/core.h"
#include <algorithm>
#include <execution>
namespace pd{namespace vision{

    template<typename Warp>
    LukasKanadeInverseCompositional<Warp>::LukasKanadeInverseCompositional (const Image& templ, const MatXi& dTx, const MatXi& dTy,
     const Image& image,std::shared_ptr<Warp> w0, std::shared_ptr<const vslam::solver::Loss> l,
      double minGradient, vslam::solver::Scaler::ShPtr scaler, std::shared_ptr<const vslam::solver::Prior<Warp::nParameters>> prior)
    : vslam::solver::Problem<Warp::nParameters>()
    , _T(templ)
    , _I(image)
    , _w(w0)
    , _loss(l)
    , _scaler(scaler)
    , _prior(prior)
    , _minGradient(minGradient)
    {
        //TODO this could come from some external feature selector
        //TODO move dTx, dTy computation outside
        _interestPoints.reserve(_T.rows() *_T.cols());
        for (uint32_t v = 0; v < _T.rows(); v++)
        {
            for (uint32_t u = 0; u < _T.cols(); u++)
            {
                if( std::sqrt(dTx(v,u)*dTx(v,u)+dTy(v,u)*dTy(v,u)) >= _minGradient)
                {
                    _interestPoints.push_back({u,v});
                }
                    
            }
        }
        _J.conservativeResize(_T.rows() *_T.cols(),Eigen::NoChange);
        _J.setZero();
         Eigen::MatrixXd steepestDescent = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
        std::for_each(std::execution::par_unseq,_interestPoints.begin(),_interestPoints.end(),[&](auto kp)
            {
                const Eigen::Matrix<double, 2,nParameters> Jw = _w->J(kp.x(),kp.y());
                        
                _J.row(kp.y() * _T.cols() + kp.x()) = Jw.row(0) * dTx(kp.y(), kp.x()) + Jw.row(1) * dTy(kp.y(),kp.x());
                steepestDescent(kp.y(),kp.x()) = _J.row(kp.y() * _T.cols() + kp.x()).norm();
            }
        );

        LOG_IMG("SteepestDescent") << steepestDescent;
    }
    template<typename Warp>
    LukasKanadeInverseCompositional<Warp>::LukasKanadeInverseCompositional (const Image& templ, const Image& image,std::shared_ptr<Warp> w0, std::shared_ptr<const vslam::solver::Loss> l, double minGradient, vslam::solver::Scaler::ShPtr scaler,std::shared_ptr<const vslam::solver::Prior<Warp::nParameters>> prior)
    : LukasKanadeInverseCompositional<Warp> (templ, algorithm::gradX(templ), algorithm::gradY(templ), image, w0, l, minGradient, scaler,prior){}



    template<typename Warp>
    void LukasKanadeInverseCompositional<Warp>::updateX(const Eigen::Matrix<double,Warp::nParameters,1>& dx)
    {
        _w->updateCompositional(-dx);
    }
    // 0,0 -+-10-> -10,-10 --10-> -10,-10
    template<typename Warp>
    typename vslam::solver::NormalEquations<Warp::nParameters>::ConstShPtr LukasKanadeInverseCompositional<Warp>::computeNormalEquations() 
    {
        Image IWxp = Image::Zero(_I.rows(),_I.cols());
        std::vector<Eigen::Vector2i> interestPointsValid(_interestPoints.size());
        auto it = std::copy_if(std::execution::par_unseq,_interestPoints.begin(),_interestPoints.end(),interestPointsValid.begin(),
        [&](auto kp) {
                Eigen::Vector2d uvI = _w->apply(kp.x(),kp.y());
                if (1 < uvI.x() && uvI.x() < _I.cols() - 1  &&
                    1 < uvI.y() && uvI.y() < _I.rows() - 1)
                {
                    IWxp(kp.y(),kp.x()) =  algorithm::bilinearInterpolation(_I,uvI.x(),uvI.y());
                    return true;
                }
                return false;
            }
        );
        interestPointsValid.resize(std::distance(interestPointsValid.begin(),it));
        if(interestPointsValid.empty()) { throw std::runtime_error("No valid interest points!"); }
        
        const MatXd R = IWxp.cast<double>() - _T.cast<double>();
        
        std::vector<double> r(interestPointsValid.size());
        std::transform(std::execution::unseq,interestPointsValid.begin(),interestPointsValid.end(),r.begin(),[&](auto kp){
            return R(kp.y(),kp.x());
        });
        const Eigen::VectorXd rScaled = _scaler->scale(Eigen::Map<Eigen::VectorXd> (r.data(),r.size())); 
        
        auto ne = std::make_shared<vslam::solver::NormalEquations<Warp::nParameters>>();
        Eigen::MatrixXd W = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
        for(size_t i = 0; i < interestPointsValid.size(); i++)
        {
            const auto& kp = interestPointsValid[i];
            W(kp.y(),kp.x()) = _loss->computeWeight(rScaled(i));
            if(!std::isfinite(_J.row(kp.y() * _T.cols() + kp.x()).norm()) || !std::isfinite(R(kp.y(),kp.x())) || !std::isfinite(W(kp.y(),kp.x())))
            {
                std::stringstream ss;
                ss << "NaN during LK with: R = " << R(kp.y(),kp.x()) << " W = "<< W(kp.y(),kp.x()) << " J = " << _J.row(kp.y() * _T.cols() + kp.x());
                throw std::runtime_error(ss.str());
            }
            /* With R = I(W(p)) - T normal equations boil down to  JWJ = -JWr. Hence, we invert the jacobian here.*/
            ne->addConstraint(_J.row(kp.y() * _T.cols() + kp.x()),R(kp.y(),kp.x()),W(kp.y(),kp.x()));
        }
        ne->A.noalias() = ne->A / (double)ne->nConstraints;
        ne->b.noalias() = ne->b / (double)ne->nConstraints;

        if (_prior){ _prior->apply(ne,_w->x()); }

        LOG_IMG("ImageWarped") << IWxp;
        LOG_IMG("Residual") << R;
        LOG_IMG("Weights") << W;

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