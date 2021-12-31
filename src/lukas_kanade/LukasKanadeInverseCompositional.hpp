
#include "LukasKanadeInverseCompositional.h"

#include "utils/visuals.h"
#include "core/algorithm.h"
namespace pd{namespace vision{

    template<typename Warp>
    LukasKanadeInverseCompositional<Warp>::LukasKanadeInverseCompositional (const Image& templ, const Image& image,std::shared_ptr<Warp> w0)
    : _T(templ)
    , _Iref(image)
    , _w(w0)
    , _IWxp(Image::Zero(_Iref.rows(),_Iref.cols()))
    , _J(Eigen::MatrixXd::Zero(_Iref.rows()*_Iref.cols(),Warp::nParameters))
    , _dIx(algorithm::gradX(templ))
    , _dIy(algorithm::gradY(templ))
    {
        Eigen::MatrixXd steepestDescent = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
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
                            
                    _J.row(idxPixel) = (dIxWp(v,u) * Jwarp.row(0) + dIyWp(v,u) * Jwarp.row(1));
                    steepestDescent(v,u) = _J.row(idxPixel).norm();
                }
                idxPixel++;
                    
            }
        }
        const auto dIWxpmat = vis::drawAsImage(dIxWp.cast<double>());
        const auto dIWypmat = vis::drawAsImage(dIyWp.cast<double>());

        Log::getImageLog("Gradient X Warped")->append(dIWxpmat);
        Log::getImageLog("Gradient Y Warped")->append(dIWypmat);
        Log::getImageLog("SteepestDescent")->append(vis::drawAsImage,steepestDescent);
    }

  

    template<typename Warp>
    bool LukasKanadeInverseCompositional<Warp>::computeResidual(Eigen::VectorXd& r)
    {

        Eigen::MatrixXd residualImage = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
        Eigen::MatrixXd visibilityImage = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
        r.setZero();
        int idxPixel = 0;
        for (int v = 0; v < _T.rows(); v++)
        {
            for (int u = 0; u < _T.cols(); u++)
            {
                Eigen::Vector2d uvWarped = _w->apply(u,v);
                if (1 < uvWarped.x() && uvWarped.x() < _Iref.cols() -1  &&
                   1 < uvWarped.y() && uvWarped.y() < _Iref.rows()-1)
                {
                    _IWxp(v,u) =  algorithm::bilinearInterpolation(_Iref,uvWarped.x(),uvWarped.y());
                    r(idxPixel) = _IWxp(v,u) - _T(v,u);
                    residualImage(v,u) = r(idxPixel);
                    visibilityImage(v,u) = 1.0;
                }
                idxPixel++;
            }
        }
        const auto IWxpmat = vis::drawAsImage(_IWxp.cast<double>());
        Log::getImageLog("Image Warped")->append(IWxpmat);
        Log::getImageLog("Residual")->append(vis::drawAsImage,residualImage);
        Log::getImageLog("Visibility")->append(vis::drawAsImage,visibilityImage);
        return true;
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

 
    
}}