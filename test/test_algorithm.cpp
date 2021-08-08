//
// Created by phil on 10.10.20.
//

#include <gtest/gtest.h>
#include "utils/Exceptions.h"
#include "core/algorithm.h"
using namespace testing;
using namespace pd;
using namespace pd::vision;

TEST(MathTest,BilinearInterpolation)
{
    Eigen::Matrix3i m;
    m << 128,128,128,
    256,256,256,
    256,256,256;
    const int r = algorithm::bilinearInterpolation(m,0,0.5);
    EXPECT_EQ(r,(256+128)/2);
}


TEST(AlgorithmTest, Gradient)
{
    Eigen::Matrix3i m;
    m << 128,128,128,
            256,128,256,
            256,256,256;

    const auto ix = algorithm::gradX(m);
    const auto iy = algorithm::gradY(m);
    const auto r = algorithm::gradient(m);

    EXPECT_EQ(ix(0,0),0);
    EXPECT_EQ(ix(1,0),-128);
    EXPECT_EQ(ix(2,0),0);

    EXPECT_EQ(iy(0,0),128);
    EXPECT_EQ(iy(1,0),0);
    EXPECT_EQ(iy(2,0),0);


    EXPECT_EQ(r(0,0),128);
    EXPECT_EQ(r(1,0),128);
    EXPECT_EQ(r(2,0),0);


}
