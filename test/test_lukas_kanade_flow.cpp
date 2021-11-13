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
#include "lukas_kanade/LukasKanadeOpticalFlow.h"

using namespace testing;
using namespace pd;
using namespace pd::vision;


class LukasKanadeOpticalFlowTest : public TestWithParam<int>{
    public:
    Image img0,img1;
    Eigen::Matrix3d A;
    Eigen::Vector2d x;
    LukasKanadeOpticalFlowTest()
    {
        img0 = utils::loadImage(TEST_RESOURCE"/person.jpg",50,50,true);
        A = Eigen::Matrix3d::Identity();
        img1 = img0;
        algorithm::warpAffine(img0,A,img1);
        x << random::U(1,2)*random::sign(),random::U(1,2)*random::sign();
    
    }
};

TEST_P(LukasKanadeOpticalFlowTest,LukasKanadeOpticalFlow)
{

    auto mat0 = vis::drawMat(img0);
    auto mat1 = vis::drawMat(img1);

    Log::getImageLog("I")->append(mat0);
    Log::getImageLog("T")->append(mat1);

    auto lk = std::make_shared<LukasKanadeOpticalFlow>(img1,img0,50);
    Eigen::Matrix3d Ares = Eigen::Matrix3d::Identity();
    Ares(0,2) = x(0);
    Ares(1,2) = x(1);
    std::cout << Ares << std::endl;
    
    ASSERT_FALSE((Ares - A).norm() <= 0.5) << "Test should have minimum shift.";
   
    lk->solve(x);
    Ares(0,2) = x(0);
    Ares(1,2) = x(1);
    
   
    std::cout << Ares << std::endl;
    EXPECT_NEAR((Ares - A).norm(),0,0.5);
}
INSTANTIATE_TEST_CASE_P(Instantiation, LukasKanadeOpticalFlowTest, ::testing::Range(1, 1));


