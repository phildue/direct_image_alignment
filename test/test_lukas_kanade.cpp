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
#include "lukas_kanade/LukasKanade.h"

using namespace testing;
using namespace pd;
using namespace pd::vision;

TEST(LukasKanadeTest,LukasKanadeOpticalFlow)
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
    x(0,0) = A(0,2);
    x(1,0) = A(1,2)+4;
    lk->solve(x);

    
    Eigen::MatrixXd Ares(3,3);
    Ares.setIdentity();
    Ares(0,2) = x(0,0);
    Ares(1,2) = x(1,0);
    std::cout << Ares << std::endl;
    EXPECT_NEAR((Ares - A).norm(),0,0.5);
}
