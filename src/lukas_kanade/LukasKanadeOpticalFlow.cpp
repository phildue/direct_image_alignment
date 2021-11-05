#include "LukasKanadeOpticalFlow.h"
#include "solver/LevenbergMarquardt.h"
#include "solver/GaussNewton.h"

#include "utils/visuals.h"
namespace pd{namespace vision{

 LukasKanadeOpticalFlow::LukasKanadeOpticalFlow (const Image& templ, const Image& image, int maxIterations, double minStepSize, double minGradient)
    : _T(image)
    , _Iref(templ)
    , _dIx(algorithm::gradX(image))
    , _dIy(algorithm::gradY(image))
    /*
    , _solver(std::make_shared<LevenbergMarquardt<2>>(
                [&](const Eigen::Vector2d& x, Eigen::VectorXd& residual, Eigen::VectorXd& weights) { return this->computeResidual(x,residual,weights);},
                [&](const Eigen::Vector2d& x, Eigen::Matrix<double,Eigen::Dynamic,2>& jacobian) { return this->computeJacobian(x,jacobian);},
                [&](const Eigen::Vector2d& dx, Eigen::Vector2d& x) { return this->updateX(dx,x);},
                (templ.cols())*(templ.rows()),
                maxIterations,
                minGradient,
                minStepSize,
                1e-8,
                1e8
                )
    )*/
    , _solver(std::make_shared<GaussNewton<2>>(
            [&](const Eigen::Vector2d& x, Eigen::VectorXd& residual, Eigen::VectorXd& weights) { return this->computeResidual(x,residual,weights);},
            [&](const Eigen::Vector2d& x, Eigen::Matrix<double,Eigen::Dynamic,2>& jacobian) { return this->computeJacobian(x,jacobian);},
            [&](const Eigen::Vector2d& dx, Eigen::Vector2d& x) { return this->updateX(dx,x);},
            (templ.cols())*(templ.rows()),
            0.1,
            minStepSize,
            maxIterations
            ))
    
    {

    }
    void LukasKanadeOpticalFlow::solve(Eigen::Vector2d& x) const
    {
        return _solver->solve(x);
    }

    bool LukasKanadeOpticalFlow::computeResidual(const Eigen::Vector2d& x, Eigen::VectorXd& r, Eigen::VectorXd& w) const
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
    bool LukasKanadeOpticalFlow::computeJacobian(const Eigen::Vector2d& x, Eigen::Matrix<double,-1,2>& j) const
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

                    Eigen::Matrix2d Jwarp = Eigen::Matrix2d::Identity();
                            
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
    bool LukasKanadeOpticalFlow::updateX(const Eigen::Vector2d& dx, Eigen::Vector2d& x) const
    {
        x.noalias() += dx;

        return true;
    }

    Eigen::Vector2d LukasKanadeOpticalFlow::warp(int u, int v,const Eigen::Vector2d& x) const
    {
          Eigen::Vector2d uvWarped;
          uvWarped << u + x.x(), v + x.y();
          return uvWarped;    
    }

  
}}