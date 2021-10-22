//
// Created by phil on 10.10.20.
//

#include <gtest/gtest.h>
#include "utils/Exceptions.h"
#include "utils/Log.h"
#include "utils/utils.h"
#include "utils/visuals.h"
#include "core/algorithm.h"
#include "core/types.h"
#include "solver/LeastSquaresSolver.h"
using namespace testing;
using namespace pd;
using namespace pd::vision;

class Cost{
    const Image _T;
    const Image _Iref;
    Eigen::MatrixXi _dIx;
    Eigen::MatrixXi _dIy;
    public:
    Cost (const Image& templ, const Image& image)
    : _T(templ)
    , _Iref(image)
    , _dIx(algorithm::gradX(image))
    , _dIy(algorithm::gradY(image))
    {

    }
    //
    // r = T(x) - I(W(x,p))
    //
    bool computeResidual(const Eigen::VectorXd& x, Eigen::VectorXd& r, Eigen::VectorXd& w) const
    {
         Eigen::MatrixXd A(3,3);
        A.setIdentity();
        /*A(0,0) = x(0,0);
        A(0,1) = x(1,0);
        A(0,2) = x(2,0);
        A(1,0) = x(3,0);
        A(1,1) = x(4,0);
        A(1,2) = x(5,0);*/
         A(0,2) = x(0,0);
        A(1,2) = x(1,0);
        int idxPixel = 0;
        Eigen::MatrixXd residualImage(_T.rows(),_T.cols());
        residualImage.setZero();
        Eigen::MatrixXd weightsImage(_T.rows(),_T.cols());
        weightsImage.setZero();
        Image IWxp = _Iref;
        algorithm::warpAffine(_Iref,A,IWxp);
        r.setZero();
        w.setZero();
        for (int v = 0; v < _T.rows(); v++)
        {
            for (int u = 0; u < _T.cols(); u++)
            {
                const Eigen::Vector3d uv1(u,v,1);
                const auto pWarped = A.inverse() * uv1;
                if (1 < pWarped.x() && pWarped.x() < _Iref.cols() -1  &&
                    1 < pWarped.y() && pWarped.y() < _Iref.rows()-1)
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
    bool computeJacobian(const Eigen::VectorXd& x, Eigen::MatrixXd& j) const
    {
        //f
        // Wx = (1 + a00)*x + a01*y + b0
        // Wy = a10*x + (1+a11)*y + b1
        // dt0/x = a00
        // dt1/y = a11
        Eigen::MatrixXd A(3,3);
        A.setIdentity();
        /*A(0,0) = x(0,0);
        A(0,1) = x(1,0);
        A(0,2) = x(2,0);
        A(1,0) = x(3,0);
        A(1,1) = x(4,0);
        A(1,2) = x(5,0);*/
        A(0,2) = x(0,0);
        A(1,2) = x(1,0);
        Eigen::MatrixXd steepestDescent(_T.rows(),_T.cols());
        j.setZero();
        int idxPixel = 0;
        Eigen::MatrixXi dIxWp = _dIx,dIyWp = _dIy;
        algorithm::warpAffine(_dIx,A,dIxWp);
        algorithm::warpAffine(_dIy,A,dIyWp);

        for (int v = 0; v < _T.rows(); v++)
        {
            for (int u = 0; u < _T.cols(); u++)
            {
                /*Eigen::Matrix<double,2,6> J_Ap;
                J_Ap << u,0,v,0,1,0,
                        0,u,0,v,0,1;*/
                Eigen::Matrix<double,2,2> J_Ap;
                J_Ap << 1,0,
                        0,1;
                        
                j.row(idxPixel) = (dIxWp(v,u) * J_Ap.row(0) + dIyWp(v,u) * J_Ap.row(1));
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
    bool updateX(const Eigen::VectorXd& dx, Eigen::VectorXd& x) const
    {
        x.noalias() += dx;

        return true;
    }
};

TEST(SolverTest,SolveGaussNewton)
{
    const Image img0 = utils::loadImage(TEST_RESOURCE"/stuff.png",50,50,true);
    Eigen::Matrix<double, 3,3> A;
    A << 1, 0,  2,
         0, 1, -1,
         0, 0,  1;

    Image img1 = img0;
    algorithm::warpAffine(img0,A,img1);
    auto mat0 = vis::drawMat(img0);
    auto mat1 = vis::drawMat(img1);

    Log::getImageLog("I")->append(mat0);
    Log::getImageLog("T")->append(mat1);

    auto cost = std::make_shared<Cost>(img1,img0);
    auto lls = std::make_shared<GaussNewton>(
                [&](const Eigen::VectorXd& x, Eigen::VectorXd& residual, Eigen::VectorXd& weights) { return cost->computeResidual(x,residual,weights);},
                [&](const Eigen::VectorXd& x, Eigen::MatrixXd& jacobian) { return cost->computeJacobian(x,jacobian);},
                [&](const Eigen::VectorXd& dx, Eigen::VectorXd& x) { return cost->updateX(dx,x);},
                (img1.cols())*(img1.rows()),
                2,
                0.1,
                1e-3,
                50);
   
    /*Eigen::MatrixXd x(6,1);
    x(0,0) = A(0,0);
    x(1,0) = A(0,1);
    x(2,0) = A(0,2) + 2;
    x(3,0) = A(1,0);
    x(4,0) = A(1,1);
    x(5,0) = A(1,2);*/
    Eigen::VectorXd x(2);
    x(0,0) = A(0,2);
    x(1,0) = A(1,2)+4;
    lls->solve(x);

    
    Eigen::MatrixXd Ares(3,3);
    Ares.setIdentity();
    /*Ares(0,0) = x(0,0);
    Ares(0,1) = x(1,0);
    Ares(0,2) = x(2,0);
    Ares(1,0) = x(3,0);
    Ares(1,1) = x(4,0);
    Ares(1,2) = x(5,0);*/
    Ares(0,2) = x(0,0);
    Ares(1,2) = x(1,0);
    std::cout << Ares << std::endl;
    EXPECT_NEAR((Ares - A).norm(),0,0.5);
}

TEST(SolverTest,SolveLM)
{
    const Image img0 = utils::loadImage(TEST_RESOURCE"/stuff.png",50,50,true);
    Eigen::Matrix<double, 3,3> A;
    A << 1, 0,  2,
         0, 1, -1,
         0, 0,  1;

    Image img1 = img0;
    algorithm::warpAffine(img0,A,img1);
    auto mat0 = vis::drawMat(img0);
    auto mat1 = vis::drawMat(img1);

    Log::getImageLog("I")->append(mat0);
    Log::getImageLog("T")->append(mat1);

    auto cost = std::make_shared<Cost>(img1,img0);
    auto lls = std::make_shared<LeastSquaresSolver>(
                [&](const Eigen::VectorXd& x, Eigen::VectorXd& residual, Eigen::VectorXd& weights) { return cost->computeResidual(x,residual,weights);},
                [&](const Eigen::VectorXd& x, Eigen::MatrixXd& jacobian) { return cost->computeJacobian(x,jacobian);},
                [&](const Eigen::VectorXd& dx, Eigen::VectorXd& x) { return cost->updateX(dx,x);},
                (img1.cols())*(img1.rows()),
                2,
                50,
                1e-3,
                1e-3);
   
    /*Eigen::MatrixXd x(6,1);
    x(0,0) = A(0,0);
    x(1,0) = A(0,1);
    x(2,0) = A(0,2) + 2;
    x(3,0) = A(1,0);
    x(4,0) = A(1,1);
    x(5,0) = A(1,2);*/
    Eigen::VectorXd x(2,1);
    x(0,0) = A(0,2)+3;
    x(1,0) = A(1,2)+4;
    lls->solve(x);

    
    Eigen::MatrixXd Ares(3,3);
    Ares.setIdentity();
    /*Ares(0,0) = x(0,0);
    Ares(0,1) = x(1,0);
    Ares(0,2) = x(2,0);
    Ares(1,0) = x(3,0);
    Ares(1,1) = x(4,0);
    Ares(1,2) = x(5,0);*/
    Ares(0,2) = x(0,0);
    Ares(1,2) = x(1,0);
    std::cout << Ares << std::endl;
    EXPECT_NEAR((Ares - A).norm(),0,0.5);
}
