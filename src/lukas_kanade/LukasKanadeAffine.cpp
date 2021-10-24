#include "LukasKanadeAffine.h"
#include "solver/LevenbergMarquardt.h"
#include "utils/visuals.h"
namespace pd{namespace vision{

    LukasKanadeAffine::LukasKanadeAffine (const Image& templ, const Image& image, int maxIterations, double minStepSize, double minGradient)
    : _T(templ)
    , _Iref(image)
    , _dIx(algorithm::gradX(image))
    , _dIy(algorithm::gradY(image))
    , _solver(std::make_shared<LevenbergMarquardt<6>>(
                [&](const Eigen::Vector6d& x, Eigen::VectorXd& residual, Eigen::VectorXd& weights) { return this->computeResidual(x,residual,weights);},
                [&](const Eigen::Vector6d& x, Eigen::Matrix<double,Eigen::Dynamic,6>& jacobian) { return this->computeJacobian(x,jacobian);},
                [&](const Eigen::Vector6d& dx, Eigen::Vector6d& x) { return this->updateX(dx,x);},
                (templ.cols())*(templ.rows()),
                maxIterations,
                minGradient,
                minStepSize)
    )
    {

    }

     void LukasKanadeAffine::solve(Eigen::Vector6d& x) const
    {
        return _solver->solve(x);
    }

    bool LukasKanadeAffine::computeResidual(const Eigen::Vector6d& x, Eigen::VectorXd& r, Eigen::VectorXd& w) const
    {
        Eigen::Matrix3d warp;
        warp << 1+x(0),   x(2), x(4),
                  x(1), 1+x(3), x(5),
                     0,      0,    1;

        Eigen::MatrixXd residualImage = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
        Eigen::MatrixXd weightsImage = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
        Image IWxp = _Iref;
        algorithm::warpAffine(_Iref,warp,IWxp);
        r.setZero();
        w.setZero();
        const double cx = _T.cols()/2;
        const double cy = _T.rows()/2;
        const Eigen::Matrix3d warpInv = warp.inverse();
        int idxPixel = 0;
        for (int v = 0; v < _T.rows(); v++)
        {
            for (int u = 0; u < _T.cols(); u++)
            {
                const Eigen::Vector3d xy1(u - cx,v - cy,1);
                const auto xy1Ref = warp.inverse() * xy1;
                const auto uv1Ref = xy1Ref + Eigen::Vector3d(cx,cy,0);
                if (1 < uv1Ref.x() && uv1Ref.x() < _Iref.cols() -1  &&
                1 < uv1Ref.y() && uv1Ref.y() < _Iref.rows()-1)
                {
                    r(idxPixel) = IWxp(v,u) - _T(v,u);
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
    bool LukasKanadeAffine::computeJacobian(const Eigen::Vector6d& x, Eigen::Matrix<double, -1,6>& j) const
    {
        Eigen::Matrix3d warp;
        warp << 1+x(0),   x(2), x(4),
                  x(1), 1+x(3), x(5),
                     0,      0,    1;
       Eigen::MatrixXd steepestDescent = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
        j.setZero();
        Eigen::MatrixXi dIxWp = _dIx,dIyWp = _dIy;
        algorithm::warpAffine(_dIx,warp,dIxWp);
        algorithm::warpAffine(_dIy,warp,dIyWp);
        int idxPixel = 0;
        const double cx = _T.cols()/2;
        const double cy = _T.rows()/2;
       for (int v = 0; v < _T.rows(); v++)
        {
            for (int u = 0; u < _T.cols(); u++)
            {
                Eigen::Matrix<double,2,6> Jwarp;
                Jwarp << u - cx,0,v - cy,0,1,0,
                         0,u - cx,0,v - cy,0,1;
                        
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
    bool LukasKanadeAffine::updateX(const Eigen::Vector6d& dx, Eigen::Vector6d& x) const
    {
        x.noalias() += dx;
        /*Eigen::Matrix3d warp;
        warp << x(0),x(1),x(2),
                x(3),x(4),x(5),
                0   ,   0,  1;

        Eigen::Matrix3d dwarp;
        dwarp << dx(0),dx(1),dx(2),
                dx(3),dx(4),dx(5),
                0   ,   0,  1;
        warp = warp * dwarp;
        x(0) = warp(0,0);
        x(1) = warp(0,1);
        x(2) = warp(0,2);
        x(3) = warp(1,0);
        x(4) = warp(1,1);
        x(5) = warp(1,2);*/

        return true;
    }
}}