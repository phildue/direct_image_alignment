
#include "LukasKanade.h"

#include "utils/visuals.h"
#include "core/algorithm.h"
#include "utils/Log.h"
namespace pd{namespace vision{

    template<typename Warp>
    LukasKanade<Warp>::LukasKanade (const Image& templ, const Image& image,std::shared_ptr<Warp> w0, std::shared_ptr<Loss> l)
    : _T(templ)
    , _Iref(image)
    , _dIx(algorithm::gradX(image))
    , _dIy(algorithm::gradY(image))
    , _w(w0)
    , _l(l)
    {}

    template<typename Warp>
    void LukasKanade<Warp>::computeResidual(Eigen::VectorXd& r, Eigen::VectorXd& w)
    {
        r.conservativeResize(_T.rows()*_T.cols());
        w.conservativeResize(_T.rows()*_T.cols());

        Eigen::MatrixXd residuals = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
        Eigen::MatrixXd visibilityImage = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
        Image IWxp = Image::Zero(_Iref.rows(),_Iref.cols());
        int idxPixel = 0;
        std::vector<double> validRs;
        validRs.reserve(_T.rows()*_T.cols());
        for (int v = 0; v < _T.rows(); v++)
        {
            for (int u = 0; u < _T.cols(); u++)
            {
                Eigen::Vector2d uvWarped = _w->apply(u,v);
                if (!std::isinf(uvWarped.norm()) && !std::isnan(uvWarped.norm()) &&
                    1 < uvWarped.x() && uvWarped.x() < _Iref.cols() -1  &&
                   1 < uvWarped.y() && uvWarped.y() < _Iref.rows()-1)
                {
                    IWxp(v,u) =  algorithm::bilinearInterpolation(_Iref,uvWarped.x(),uvWarped.y());
                    residuals(v,u) = _T(v,u) - IWxp(v,u);
                    visibilityImage(v,u) = 1.0;
                    r(idxPixel) = residuals(v,u);
                    validRs.push_back(r(idxPixel));
                }else{
                    w(idxPixel) = 0.0;
                    r(idxPixel) = 0.0;
                }
                idxPixel++;
            }
        }
        const double median = algorithm::median(validRs);
        const Eigen::VectorXd rScaled = median != 0 ? (r.array() - median).array()/median : r;
        idxPixel = 0;
        for (int v = 0; v < _T.rows(); v++)
        {
            for (int u = 0; u < _T.cols(); u++)
            {
                if (visibilityImage(v,u) > 0.0)
                {
                    w(idxPixel) = _l->computeWeight(rScaled(idxPixel));
                    visibilityImage(v,u) = w(idxPixel);
                }
                idxPixel++;
            }
        }
        LOG_IMG("ImageWarped") << IWxp;
        LOG_IMG("Residual") << residuals;
        LOG_IMG("Visibility") << visibilityImage;
    }
    //
    // J = Ixy*dW/dp
    //
    template<typename Warp>
    bool LukasKanade<Warp>::computeJacobian(Eigen::Matrix<double, -1,Warp::nParameters>& j)
    {
        Eigen::MatrixXd steepestDescent = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
        j.setZero();
        Eigen::MatrixXi dIxWp = Eigen::MatrixXi::Zero(_Iref.rows(),_Iref.cols());
        Eigen::MatrixXi dIyWp = Eigen::MatrixXi::Zero(_Iref.rows(),_Iref.cols());

        int idxPixel = 0;
        for (int v = 0; v < _T.rows(); v++)
        {
            for (int u = 0; u < _T.cols(); u++)
            {
                Eigen::Vector2d uvWarped = _w->apply(u,v);
                if (1 < uvWarped.x() && uvWarped.x() < _Iref.cols() -1  &&
                   1 < uvWarped.y() && uvWarped.y() < _Iref.rows()-1)
                {
                    dIxWp(v,u) = algorithm::bilinearInterpolation(_dIx,uvWarped.x(),uvWarped.y());
                    dIyWp(v,u) = algorithm::bilinearInterpolation(_dIy,uvWarped.x(),uvWarped.y());

                    const Eigen::Matrix<double, 2,nParameters> Jwarp = _w->J(u,v);
                            
                    j.row(idxPixel++) = (dIxWp(v,u) * Jwarp.row(0) + dIyWp(v,u) * Jwarp.row(1));
                    steepestDescent(v,u) = j.row(idxPixel).norm();
                }
                    
            }
        }

        LOG_IMG("Gradient_X_Warped") << dIxWp;
        LOG_IMG("Gradient_Y_Warped") << dIyWp;
        LOG_IMG("SteepestDescent") << steepestDescent;
        return true;
    }

    template<typename Warp>
    bool LukasKanade<Warp>::updateX(const Eigen::Matrix<double,Warp::nParameters,1>& dx)
    {
        _w->updateAdditive(dx);
        return true;
    }


    template<typename Warp>
    void LukasKanade<Warp>::extendLeft(Eigen::MatrixXd& H)
    {}
    template<typename Warp>
    void LukasKanade<Warp>::extendRight(Eigen::VectorXd& g)
    {}
}}