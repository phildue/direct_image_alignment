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
#include "lukas_kanade/LukasKanadeAffine.h"
#include "lukas_kanade/LukasKanadeOpticalFlow.h"

using namespace testing;
using namespace pd;
using namespace pd::vision;

TEST(LukasKanadeTest,DISABLED_LukasKanadeOpticalFlow)
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

    auto lk = std::make_shared<LukasKanadeOpticalFlow>(img1,img0,50);
   
    Eigen::Vector2d x(2);
    x(0) = A(0,2)+2;
    x(1) = A(1,2)+2;
    lk->solve(x);

    
    Eigen::MatrixXd Ares(3,3);
    Ares.setIdentity();
    Ares(0,2) = x(0);
    Ares(1,2) = x(1);
    std::cout << Ares << std::endl;
    EXPECT_NEAR((Ares - A).norm(),0,0.5);
}

TEST(LukasKanadeTest,LukasKanadeAffine)
{
    const Image img0 = utils::loadImage(TEST_RESOURCE"/stuff.png",50,50,true);
    const Eigen::Matrix<double, 3,3> A = transforms::createdTransformMatrix2D(1,1,0);

    Image img1 = Image::Zero(img0.rows(),img0.cols());
    algorithm::warpAffine(img0,A,img1);
    auto mat0 = vis::drawMat(img0);
    auto mat1 = vis::drawMat(img1);

    Log::getImageLog("I")->append(mat0);
    Log::getImageLog("T")->append(mat1);

    auto lk = std::make_shared<LukasKanadeAffine>(img1,img0,50,1e-5);
   
    const Eigen::Matrix<double, 3,3> Anoisy = transforms::createdTransformMatrix2D(2,2,0.2);
    Eigen::Vector6d x = Eigen::Vector6d::Zero();
    x(0) = Anoisy(0,0)-1;
    x(1) = Anoisy(1,0);
    x(2) = Anoisy(0,1);
    x(3) = Anoisy(1,1)-1;
    x(4) = Anoisy(0,2);
    x(5) = Anoisy(1,2);
    
    Eigen::Matrix3d Ares;
    Ares << 1+x(0),   x(2), x(4),
              x(1), 1+x(3), x(5),
                 0,      0,    1;
    EXPECT_FALSE((Ares - A).norm() <= 0.5);

    lk->solve(x);
    
    Ares << 1+x(0),   x(2), x(4),
              x(1), 1+x(3), x(5),
                 0,      0,    1;
 
    std::cout << "A:\n" << A << "Ares:\n"<< Ares << std::endl;
    EXPECT_NEAR((Ares - A).norm(),0,0.5);
}
