
#include "LukasKanade.h"
#include "solver/LevenbergMarquardt.h"
#include "solver/GaussNewton.h"

#include "utils/visuals.h"
#include "core/algorithm.h"
namespace pd{namespace vision{

    template<int nParameters>
    LukasKanade<nParameters>::LukasKanade (const Image& templ, const Image& image, int maxIterations, double minStepSize, double minGradient)
    : _T(templ)
    , _Iref(image)
    , _dIx(algorithm::gradX(image))
    , _dIy(algorithm::gradY(image))
    /*, _solver(std::make_shared<LevenbergMarquardt<nParameters>>(
                [&](const Eigen::Matrix<double,nParameters,1>& x, Eigen::VectorXd& residual, Eigen::VectorXd& weights) { return this->computeResidual(x,residual,weights);},
                [&](const Eigen::Matrix<double,nParameters,1>& x, Eigen::Matrix<double,-1,nParameters>& jacobian) { return this->computeJacobian(x,jacobian);},
                [&](const Eigen::Matrix<double,nParameters,1>& dx, Eigen::Matrix<double, nParameters,1>& x) { return this->updateX(dx,x);},
                (templ.cols())*(templ.rows()),
                maxIterations,
                minGradient,
                minStepSize)
    )*/
     , _solver(std::make_shared<GaussNewton<nParameters>>(
                [&](const Eigen::Matrix<double,nParameters,1>& x, Eigen::VectorXd& residual, Eigen::VectorXd& weights) { return this->computeResidual(x,residual,weights);},
                [&](const Eigen::Matrix<double,nParameters,1>& x, Eigen::Matrix<double,-1,nParameters>& jacobian) { return this->computeJacobian(x,jacobian);},
                [&](const Eigen::Matrix<double,nParameters,1>& dx, Eigen::Matrix<double, nParameters,1>& x) { return this->updateX(dx,x);},
                (templ.cols())*(templ.rows()),
                0.1,
                minStepSize,
                maxIterations)
    )
    {

    }

    template<int nParameters>
    void LukasKanade<nParameters>::solve(Eigen::Matrix<double,nParameters,1>& x)
    {
        return _solver->solve(x);
    }

    template<int nParameters>
    bool LukasKanade<nParameters>::computeResidual(const Eigen::Matrix<double,nParameters,1>& x, Eigen::VectorXd& r, Eigen::VectorXd& w)
    {

        Eigen::MatrixXd residualImage = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
        Eigen::MatrixXd weightsImage = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
        Image IWxp = Image::Zero(_Iref.rows(),_Iref.cols());
        r.setZero();
        w.setZero();
        int idxPixel = 0;
        for (int v = 0; v < _T.rows(); v++)
        {
            for (int u = 0; u < _T.cols(); u++)
            {
                Eigen::Vector2d uvWarped = warp(u,v,x);
                if (1 < uvWarped.x() && uvWarped.x() < _Iref.cols() -1  &&
                   1 < uvWarped.y() && uvWarped.y() < _Iref.rows()-1)
                {
                    IWxp(v,u) =  algorithm::bilinearInterpolation(_Iref,uvWarped.x(),uvWarped.y());
                    r(idxPixel) = _T(v,u) - IWxp(v,u);
                    residualImage(v,u) = r(idxPixel);
                    w(idxPixel) = 1.0;
                    weightsImage(v,u) = w(idxPixel);
                }
                idxPixel++;
            }
        }
        const auto IWxpmat = vis::drawAsImage(IWxp.cast<double>());
        Log::getImageLog("Image Warped")->append(IWxpmat);
        Log::getImageLog("Residual")->append(vis::drawAsImage,residualImage);
        Log::getImageLog("Weights")->append(vis::drawAsImage,weightsImage);
        return true;
    }

    //
    // J = Ixy*dW/dp
    //
    template<int nParameters>
    bool LukasKanade<nParameters>::computeJacobian(const Eigen::Matrix<double,nParameters,1>& x, Eigen::Matrix<double, -1,nParameters>& j)
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
                Eigen::Vector2d uvWarped = warp(u,v,x);
                if (1 < uvWarped.x() && uvWarped.x() < _Iref.cols() -1  &&
                   1 < uvWarped.y() && uvWarped.y() < _Iref.rows()-1)
                {
                    dIxWp(v,u) = algorithm::bilinearInterpolation(_dIx,uvWarped.x(),uvWarped.y());
                    dIyWp(v,u) = algorithm::bilinearInterpolation(_dIy,uvWarped.x(),uvWarped.y());

                    const Eigen::Matrix<double, -1,nParameters> Jwarp = jacobianWarp(v,u);
                            
                    j.row(idxPixel) = (dIxWp(v,u) * Jwarp.row(0) + dIyWp(v,u) * Jwarp.row(1));
                    steepestDescent(v,u) = j.row(idxPixel).norm();
                }
                idxPixel++;
                    
            }
        }
        const auto dIWxpmat = vis::drawAsImage(dIxWp.cast<double>());
        const auto dIWypmat = vis::drawAsImage(dIyWp.cast<double>());

        Log::getImageLog("Gradient X Warped")->append(dIWxpmat);
        Log::getImageLog("Gradient Y Warped")->append(dIWypmat);
        Log::getImageLog("SteepestDescent")->append(vis::drawAsImage,steepestDescent);
        return true;
    }
    
}}