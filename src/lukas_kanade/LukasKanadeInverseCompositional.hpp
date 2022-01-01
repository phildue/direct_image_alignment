
#include "LukasKanadeInverseCompositional.h"

#include "utils/visuals.h"
#include "core/algorithm.h"

//Debug monitors
#define STEEPEST_DESCENT Log::getImageLog("SteepestDescent",Level::Debug)
#define DTX Log::getImageLog("dTx",Level::Debug)
#define DTY Log::getImageLog("dTy",Level::Debug)
#define RESIDUAL Log::getImageLog("Residual",Level::Debug)
#define IMAGE_WARPED Log::getImageLog("Image Warped",Level::Debug)
#define VISIBILITY Log::getImageLog("Visibility",Level::Debug)

namespace pd{namespace vision{

    template<typename Warp>
    LukasKanadeInverseCompositional<Warp>::LukasKanadeInverseCompositional (const Image& templ, const Image& image,std::shared_ptr<Warp> w0)
    : _T(templ)
    , _Iref(image)
    , _w(w0)
    , _J(Eigen::MatrixXd::Zero(_Iref.rows()*_Iref.cols(),Warp::nParameters))
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

        DTX << dTx;
        DTY << dTy;
        STEEPEST_DESCENT << steepestDescent;
    }

    template<typename Warp>
    bool LukasKanadeInverseCompositional<Warp>::computeResidual(Eigen::VectorXd& r)
    {
        r.setZero();

        Eigen::MatrixXd residualImage = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
        Eigen::MatrixXd visibilityImage = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
        Image IWxp = Image::Zero(_Iref.rows(),_Iref.cols());

        int idxPixel = 0;
        for (int v = 0; v < _T.rows(); v++)
        {
            for (int u = 0; u < _T.cols(); u++)
            {
                Eigen::Vector2d uvWarped = _w->apply(u,v);
                if (1 < uvWarped.x() && uvWarped.x() < _Iref.cols() -1  &&
                   1 < uvWarped.y() && uvWarped.y() < _Iref.rows()-1)
                {
                    IWxp(v,u) =  algorithm::bilinearInterpolation(_Iref,uvWarped.x(),uvWarped.y());
                    r(idxPixel) = IWxp(v,u) - _T(v,u);
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