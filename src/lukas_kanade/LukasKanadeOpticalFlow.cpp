#include "LukasKanadeOpticalFlow.h"
#include "solver/LevenbergMarquardt.h"
#include "utils/visuals.h"
namespace pd{namespace vision{

 LukasKanadeOpticalFlow::LukasKanadeOpticalFlow (const Image& templ, const Image& image, int maxIterations, double minStepSize, double minGradient)
    : _T(templ)
    , _Iref(image)
    , _dIx(algorithm::gradX(image))
    , _dIy(algorithm::gradY(image))
    , _solver(std::make_shared<LevenbergMarquardt<2>>(
                [&](const Eigen::Vector2d& x, Eigen::VectorXd& residual, Eigen::VectorXd& weights) { return this->computeResidual(x,residual,weights);},
                [&](const Eigen::Vector2d& x, Eigen::Matrix<double,Eigen::Dynamic,2>& jacobian) { return this->computeJacobian(x,jacobian);},
                [&](const Eigen::Vector2d& dx, Eigen::Vector2d& x) { return this->updateX(dx,x);},
                (templ.cols())*(templ.rows()),
                maxIterations,
                minGradient,
                minStepSize)
    )
    {

    }
    void LukasKanadeOpticalFlow::solve(Eigen::Vector2d& x) const
    {
        return _solver->solve(x);
    }

    bool LukasKanadeOpticalFlow::computeResidual(const Eigen::Vector2d& x, Eigen::VectorXd& r, Eigen::VectorXd& w) const
    {
        Eigen::Matrix3d warp = Eigen::Matrix3d::Identity();
        warp(0,2) = x(0);
        warp(1,2) = x(1);

        Eigen::MatrixXd residualImage = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
        Eigen::MatrixXd weightsImage = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
        Image IWxp = _Iref;
        algorithm::warpAffine(_Iref,warp,IWxp);
        r.setZero();
        w.setZero();
        const Eigen::Matrix3d warpInv = warp.inverse();
        const double cx = _T.cols()/2;
        const double cy = _T.rows()/2;
        int idxPixel = 0;
        for (int v = 0; v < _T.rows(); v++)
        {
            for (int u = 0; u < _T.cols(); u++)
            {
                const Eigen::Vector3d xy1(u - cx,v - cy,1);
                const Eigen::Vector3d xy1Ref = warp.inverse() * xy1;
                Eigen::Vector2d uvRef;
                uvRef << xy1Ref.x() + cx,xy1Ref.y() +cy;
                if (1 < uvRef.x() && uvRef.x() < _Iref.cols() -1  &&
                   1 < uvRef.y() && uvRef.y() < _Iref.rows()-1)
                {
                    r(idxPixel) = IWxp(v,u) -_T(v,u);
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
        Eigen::Matrix3d warp = Eigen::Matrix3d::Identity();
        warp(0,2) = x(0);
        warp(1,2) = x(1);

        Eigen::MatrixXd steepestDescent = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
        j.setZero();
        Eigen::MatrixXi dIxWp = _dIx,dIyWp = _dIy;
        algorithm::warpAffine(_dIx,warp,dIxWp);
        algorithm::warpAffine(_dIy,warp,dIyWp);

        int idxPixel = 0;
        for (int v = 0; v < _T.rows(); v++)
        {
            for (int u = 0; u < _T.cols(); u++)
            {
                Eigen::Matrix2d Jwarp = Eigen::Matrix2d::Identity();
                        
                j.row(idxPixel) = (dIxWp(v,u) * Jwarp.row(0) + dIyWp(v,u) * Jwarp.row(1));
                steepestDescent(v,u) = j.row(idxPixel).norm();

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
  
}}