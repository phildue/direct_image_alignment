//
// Created by phil on 10.10.20.
//

#include <gtest/gtest.h>
#include <Exceptions.h>
#include "math.h"
using namespace testing;
using namespace pd;
using namespace pd::vision;

TEST(MathTest,BilinearInterpolation)
{
    Eigen::Matrix3i m;
    m << 128,128,128,
    256,265,256,
    256,256,256;
    const int r = math::bilinearInterpolation(m,0,0.5);
    EXPECT_EQ(r,(256+128)/2);
}

