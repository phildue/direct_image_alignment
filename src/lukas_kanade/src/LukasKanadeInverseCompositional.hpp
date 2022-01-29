
#include "utils/utils.h"
#include "core/core.h"

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
    , _dTxy(MatD::Zero(_Iref.rows(),_Iref.cols()))
    , _minGradient(minGradient)
    {
        Eigen::MatrixXd steepestDescent = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());

        int idxPixel = 0;
        for (int v = 0; v < _T.rows(); v++)
        {
            for (int u = 0; u < _T.cols(); u++)
            {
                _dTxy(v,u) = std::sqrt(_dTx(v,u)*_dTx(v,u)+_dTy(v,u)*_dTy(v,u));
                if( _dTxy(v,u) >= _minGradient)
                {
                    const Eigen::Matrix<double, 2,nParameters> Jwarp = _w->J(u,v);
                            
                    _J.row(idxPixel) = (_dTx(v,u) * Jwarp.row(0) + _dTy(v,u) * Jwarp.row(1));
                    steepestDescent(v,u) = _J.row(idxPixel).norm();
                    idxPixel++;
                }
                    
            }
        }
        _J.conservativeResize(idxPixel, Eigen::NoChange);

        LOG_IMG("DTX") << _dTx;
        LOG_IMG("DTY") << _dTy;
        LOG_IMG("SteepestDescent") << steepestDescent;
    }

    template<typename Warp>
    void LukasKanadeInverseCompositional<Warp>::computeResidual(Eigen::VectorXd& r, Eigen::VectorXd& w)
    {

        r.conservativeResize(_T.rows()*_T.cols());
        w.conservativeResize(_T.rows()*_T.cols());

        Eigen::MatrixXd rImg = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
        Eigen::MatrixXd wImg = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
        Image IWxp = Image::Zero(_Iref.rows(),_Iref.cols());
        std::vector<double> validRs;
        w.setZero();
        r.setZero();
        validRs.reserve(_T.rows()*_T.cols());

        int idxPixel = 0;
        for (int v = 0; v < _T.rows(); v++)
        {
            for (int u = 0; u < _T.cols(); u++)
            {
                if( _dTxy(v,u) >= _minGradient)
                {
                    Eigen::Vector2d uvWarped = _w->apply(u,v);
                    if (1 < uvWarped.x() && uvWarped.x() < _Iref.cols() -1  &&
                    1 < uvWarped.y() && uvWarped.y() < _Iref.rows()-1)
                    {
                        // TODO just fill images and reshape at the end?
                        IWxp(v,u) =  algorithm::bilinearInterpolation(_Iref,uvWarped.x(),uvWarped.y());
                        r(idxPixel) = IWxp(v,u) - _T(v,u);
                        rImg(v,u) = r(idxPixel);
                        wImg(v,u) = 1.0;
                        validRs.push_back(r(idxPixel));
                    }
                    idxPixel++;
                }
            }
        }
        const Eigen::Map<Eigen::VectorXd> rValid(validRs.data(),validRs.size());
        double median = algorithm::median(rValid,true);
        const auto stddev = (rValid.array() - median).array().abs().sum()/(rValid.rows() - 1);
        const Eigen::VectorXd rScaled = (r.array() - median)/stddev;

        idxPixel = 0;
        for (int v = 0; v < _T.rows(); v++)
        {
            for (int u = 0; u < _T.cols(); u++)
            {
                if( _dTxy(v,u) >= _minGradient)
                {
                    if (wImg(v,u) > 0.0)
                    {
                        w(idxPixel) = _l->computeWeight(rScaled(idxPixel));
                        wImg(v,u) = w(idxPixel);
                    }
                    idxPixel++;
                }
            }
        }
        r.conservativeResize(idxPixel);
        w.conservativeResize(idxPixel);

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
    void LukasKanadeInverseCompositional<Warp>::extendLeft(Eigen::MatrixXd& H)
    {}
    template<typename Warp>
    void LukasKanadeInverseCompositional<Warp>::extendRight(Eigen::VectorXd& g)
    {}

}}