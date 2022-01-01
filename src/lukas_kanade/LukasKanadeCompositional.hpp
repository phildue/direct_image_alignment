
#include "LukasKanadeCompositional.h"

#include "utils/visuals.h"
#include "core/algorithm.h"
namespace pd{namespace vision{

    template<typename Warp>
    LukasKanadeCompositional<Warp>::LukasKanadeCompositional (const Image& templ, const Image& image,std::shared_ptr<Warp> w0)
    : _T(templ)
    , _Iref(image)
    , _w(w0)
    , _IWxp(Image::Zero(_Iref.rows(),_Iref.cols()))
    {
        _J.resize(_Iref.rows()*_Iref.cols());
        auto x = _w->x();
        int idxPixel = 0;
        //_w->setX(Eigen::Matrix<double,1,Warp::nParameters>::Zero());//we have to do this? but otherwise its not the identity warp
        for (int v = 0; v < _T.rows(); v++)
        {
            for (int u = 0; u < _T.cols(); u++)
            {
                Eigen::Vector2d uvWarped = _w->apply(u,v);
                if (1 < uvWarped.x() && uvWarped.x() < _Iref.cols() -1  &&
                   1 < uvWarped.y() && uvWarped.y() < _Iref.rows()-1)
                {
                    _J[idxPixel++] = _w->J(u,v);
                            
                }
                    
            }
        }
        _w->setX(x);
    }

  

    template<typename Warp>
    bool LukasKanadeCompositional<Warp>::computeResidual(Eigen::VectorXd& r)
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
                    r(idxPixel) = _T(v,u) - _IWxp(v,u);
                    residualImage(v,u) = r(idxPixel);
                    visibilityImage(v,u) = 1.0;
                }
                idxPixel++;
            }
        }
        Log::getImageLog("Image Warped")->append(_IWxp);
        Log::getImageLog("Residual")->append(residualImage);
        Log::getImageLog("Visibility")->append(visibilityImage);
        return true;
    }

    //
    // J = Ixy*dW/dp
    //
    template<typename Warp>
    bool LukasKanadeCompositional<Warp>::computeJacobian(Eigen::Matrix<double, -1,Warp::nParameters>& j)
    {
        Eigen::MatrixXd steepestDescent = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
        j.setZero();
        const Eigen::MatrixXi dIxWp = algorithm::gradX(_IWxp);
        const Eigen::MatrixXi dIyWp = algorithm::gradY(_IWxp);

        int idxPixel = 0;
        for (int v = 0; v < _T.rows(); v++)
        {
            for (int u = 0; u < _T.cols(); u++)
            {
                Eigen::Vector2d uvWarped = _w->apply(u,v);
                if (1 < uvWarped.x() && uvWarped.x() < _Iref.cols() -1  &&
                   1 < uvWarped.y() && uvWarped.y() < _Iref.rows()-1)
                {

                    const Eigen::Matrix<double, 2,nParameters> Jwarp = _J[idxPixel];
                            
                    j.row(idxPixel) = (dIxWp(v,u) * Jwarp.row(0) + dIyWp(v,u) * Jwarp.row(1));
                    steepestDescent(v,u) = j.row(idxPixel).norm();
                }
                idxPixel++;
                    
            }
        }

        Log::getImageLog("Gradient X Warped")->append(dIxWp);
        Log::getImageLog("Gradient Y Warped")->append(dIyWp);
        Log::getImageLog("SteepestDescent")->append(steepestDescent);
        return true;
    }

    template<typename Warp>
    bool LukasKanadeCompositional<Warp>::updateX(const Eigen::Matrix<double,Warp::nParameters,1>& dx)
    {
        _w->updateCompositional(dx);
        return true;
    }

 
    
}}