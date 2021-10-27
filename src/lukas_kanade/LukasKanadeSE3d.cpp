#include "LukasKanadeSE3d.h"
#include "solver/LevenbergMarquardt.h"
#include "utils/visuals.h"
namespace pd{namespace vision{

    LukasKanadeSE3d::LukasKanadeSE3d (const Image& templ, const Image& image,const Eigen::MatrixXd& depth, std::shared_ptr<Camera> cam, int maxIterations, double minStepSize, double minGradient)
    : _T(templ)
    , _Iref(image)
    , _depth(depth)
    , _cam(cam)
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

     void LukasKanadeSE3d::solve(Eigen::Vector6d& x) const
    {
        return _solver->solve(x);
    }

    bool LukasKanadeSE3d::computeResidual(const Eigen::Vector6d& x, Eigen::VectorXd& r, Eigen::VectorXd& w) const
    {
        const SE3d se3d = Sophus::SE3d::exp(x);
        const SE3d se3dInv = se3d.inverse();

        Eigen::MatrixXd residualImage = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
        Eigen::MatrixXd weightsImage = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
        r.setZero();
        w.setZero();
        int idxPixel = 0;
        for (int v = 0; v < _T.rows(); v++)
        {
            for (int u = 0; u < _T.cols(); u++)
            {
                const Eigen::Vector3d pCcs = _cam->image2camera({u,v},_depth(v,u));
                const Eigen::Vector2d uvRef = _cam->camera2image(se3dInv * pCcs);
                if (1 < uvRef.x() && uvRef.x() < _Iref.cols() -1  &&
                    1 < uvRef.y() && uvRef.y() < _Iref.rows()-1)
                {
                    r(idxPixel) = algorithm::bilinearInterpolation(_Iref, uvRef.x(),uvRef.y()) - _T(v,u);
                    residualImage(v,u) = r(idxPixel);
                    w(idxPixel) = 1.0;
                    weightsImage(v,u) = w(idxPixel);
                }
                idxPixel++;
            }
        }
        Log::getImageLog("Residual")->append(vis::drawAsImage,residualImage);
        Log::getImageLog("Weights")->append(vis::drawAsImage,weightsImage);
        return true;
    }

    //
    // J = Ixy*dW/dp
    //
    bool LukasKanadeSE3d::computeJacobian(const Eigen::Vector6d& x, Eigen::Matrix<double, -1,6>& j) const
    {
        const SE3d se3d = Sophus::SE3d::exp(x);
        const SE3d se3dInv = se3d.inverse();
        Eigen::MatrixXd steepestDescent = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
        j.setZero();
        int idxPixel = 0;
        const double cx = _T.cols()/2;
        const double cy = _T.rows()/2;
        for (int v = 0; v < _T.rows(); v++)
        {
            for (int u = 0; u < _T.cols(); u++)
            {
                const Eigen::Vector3d pCcsRef = se3dInv * _cam->image2camera({u,v},_depth(v,u));
                const Eigen::Vector2d uvRef = _cam->camera2image(pCcsRef);
                const Eigen::Matrix<double,2,6> Jwarp = _cam->J_xyz2uv(pCcsRef);  
                if (1 < uvRef.x() && uvRef.x() < _Iref.cols() -1  &&
                1 < uvRef.y() && uvRef.y() < _Iref.rows()-1)
                {
                    const double du = algorithm::bilinearInterpolation(_dIx,uvRef.x(),uvRef.y());
                    const double dv = algorithm::bilinearInterpolation(_dIy,uvRef.x(),uvRef.y());

                    j.row(idxPixel) = (du * Jwarp.row(0) + dv * Jwarp.row(1));
                    steepestDescent(v,u) = j.row(idxPixel).norm();
                }
                idxPixel++;
                    
            }
        }
        Log::getImageLog("SteepestDescent")->append(vis::drawAsImage,steepestDescent);

        return true;
    }
    bool LukasKanadeSE3d::updateX(const Eigen::Vector6d& dx, Eigen::Vector6d& x) const
    {
        x.noalias() = (Sophus::SE3d::exp(x) * Sophus::SE3d::exp(dx)).log();

        return true;
    }
}}