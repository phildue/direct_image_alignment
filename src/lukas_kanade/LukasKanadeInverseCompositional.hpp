
#include "utils/visuals.h"
#include "core/algorithm.h"
#include "utils/Log.h"

namespace pd{namespace vision{

    template<typename Warp>
    LukasKanadeInverseCompositional<Warp>::LukasKanadeInverseCompositional (const Image& templ, const Image& image,std::shared_ptr<Warp> w0, std::shared_ptr<Loss> l)
    : _T(templ)
    , _Iref(image)
    , _w(w0)
    , _J(Eigen::MatrixXd::Zero(_Iref.rows()*_Iref.cols(),Warp::nParameters))
    , _l(l)
    {
        Eigen::MatrixXd steepestDescent = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
        const Eigen::MatrixXi dTx = algorithm::gradX(templ);
        const Eigen::MatrixXi dTy = algorithm::gradY(templ);

        int idxPixel = 0;
        for (int v = 0; v < _T.rows(); v++)
        {
            for (int u = 0; u < _T.cols(); u++)
            {
                
                const Eigen::Matrix<double, 2,nParameters> Jwarp = _w->J(u,v);
                        
                _J.row(idxPixel) = (dTx(v,u) * Jwarp.row(0) + dTy(v,u) * Jwarp.row(1));
                steepestDescent(v,u) = _J.row(idxPixel).norm();
                idxPixel++;
                    
            }
        }

        LOG_IMAGE_DEBUG("DTX") << dTx;
        LOG_IMAGE_DEBUG("DTY") << dTy;
        LOG_IMAGE_DEBUG("SteepestDescent") << steepestDescent;
    }

    template<typename Warp>
    void LukasKanadeInverseCompositional<Warp>::computeResidual(Eigen::VectorXd& r, Eigen::VectorXd& w)
    {
        TIMED_SCOPE(timerI,"computeResidual");

        r.conservativeResize(_T.rows()*_T.cols());
        w.conservativeResize(_T.rows()*_T.cols());

        Eigen::MatrixXd rImg = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
        Eigen::MatrixXd wImg = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
        Image IWxp = Image::Zero(_Iref.rows(),_Iref.cols());
        std::vector<double> validRs;
        validRs.reserve(_T.rows()*_T.cols());
        {
            TIMED_SCOPE(timerI,"computeResidualLoop");

            int idxPixel = 0;
            for (int v = 0; v < _T.rows(); v++)
            {
                for (int u = 0; u < _T.cols(); u++)
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
                    }else{
                        w(idxPixel) = 0.0;
                        r(idxPixel) = 0.0;
                    }
                    idxPixel++;
                }
            }
        }
        const Eigen::Map<Eigen::VectorXd> rValid(validRs.data(),validRs.size());
        double median = algorithm::median(rValid);
        const auto stddev = (rValid.array() - median).array().abs().sum()/(rValid.rows() - 1);
        const Eigen::VectorXd rScaled = (r.array() - median)/stddev;

        {
            TIMED_SCOPE(timerI,"computeWeightsLoop");

            int idxPixel = 0;
            for (int v = 0; v < _T.rows(); v++)
            {

                for (int u = 0; u < _T.cols(); u++)
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
        LOG_IMAGE_DEBUG("ImageWarped") << IWxp;
        LOG_IMAGE_DEBUG("Residual") << rImg;
        LOG_IMAGE_DEBUG("Weights") << wImg;
    }

    //
    // J = Ixy*dW/dp
    //
    template<typename Warp>
    bool LukasKanadeInverseCompositional<Warp>::computeJacobian(Eigen::Matrix<double, -1,Warp::nParameters>& j)
    {
        TIMED_SCOPE(timerI,"computeJacobian");

        j = _J;
        return true;
    }

    

    template<typename Warp>
    bool LukasKanadeInverseCompositional<Warp>::updateX(const Eigen::Matrix<double,Warp::nParameters,1>& dx)
    {
        _w->updateCompositional(-dx);
        return true;
    }
}}