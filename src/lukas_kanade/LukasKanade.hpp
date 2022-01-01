
#include "LukasKanade.h"
#include "solver/LevenbergMarquardt.h"
#include "solver/GaussNewton.h"

#include "utils/visuals.h"
#include "core/algorithm.h"

namespace pd{namespace vision{

//Debug monitors
#define STEEPEST_DESCENT Log::getImageLog("SteepestDescent",Level::Debug)
#define GRAD_X_WARPED Log::getImageLog("Gradient X Warped",Level::Debug)
#define GRAD_Y_WARPED Log::getImageLog("Gradient Y Warped",Level::Debug)
#define RESIDUAL Log::getImageLog("Residual",Level::Debug)
#define IMAGE_WARPED Log::getImageLog("Image Warped",Level::Debug)
#define VISIBILITY Log::getImageLog("Visibility",Level::Debug)


    template<typename Warp>
    LukasKanade<Warp>::LukasKanade (const Image& templ, const Image& image,std::shared_ptr<Warp> w0)
    : _T(templ)
    , _Iref(image)
    , _dIx(algorithm::gradX(image))
    , _dIy(algorithm::gradY(image))
    , _w(w0)
    {}

    template<typename Warp>
    bool LukasKanade<Warp>::computeResidual(Eigen::VectorXd& r)
    {

        Eigen::MatrixXd residualImage = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
        Eigen::MatrixXd visibilityImage = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
        Image IWxp = Image::Zero(_Iref.rows(),_Iref.cols());
        r.setZero();
        int idxPixel = 0;
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
                    r(idxPixel) = _T(v,u) - IWxp(v,u);
                    residualImage(v,u) = r(idxPixel);
                    visibilityImage(v,u) = 1.0;
                }
                idxPixel++;
            }
        }
        IMAGE_WARPED << IWxp;
        RESIDUAL << residualImage;
        VISIBILITY << visibilityImage;
        return true;
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
                            
                    j.row(idxPixel) = (dIxWp(v,u) * Jwarp.row(0) + dIyWp(v,u) * Jwarp.row(1));
                    steepestDescent(v,u) = j.row(idxPixel).norm();
                }
                idxPixel++;
                    
            }
        }

        GRAD_X_WARPED << dIxWp;
        GRAD_Y_WARPED << dIyWp;
        STEEPEST_DESCENT << steepestDescent;
        return true;
    }

    template<typename Warp>
    bool LukasKanade<Warp>::updateX(const Eigen::Matrix<double,Warp::nParameters,1>& dx)
    {
        _w->updateAdditive(dx);
        return true;
    }
}}