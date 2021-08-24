//
// Created by phil on 10.10.20.
//

#include <gtest/gtest.h>
#include "utils/Exceptions.h"
#include "utils/Log.h"
#include "core/algorithm.h"
using namespace testing;
using namespace pd;
using namespace pd::vision;

TEST(MathTest,BilinearInterpolation)
{
    Eigen::Matrix<std::uint8_t ,3,3>m;
    m << 128,128,128,
    255,255,255,
    255,255,255;
    const int r = algorithm::bilinearInterpolation(m,0,0.5);
    EXPECT_EQ(r,(256+128)/2);
}


TEST(AlgorithmTest, Gradient)
{
    Eigen::Matrix<std::uint8_t ,3,3> m;
    m << 128,128,128,
            255,128,255,
            255,255,255;

    const auto ix = algorithm::gradX(m);
    const auto iy = algorithm::gradY(m);
    const auto r = algorithm::gradient(m);

    EXPECT_EQ(ix(0,0),0);
    EXPECT_EQ(ix(1,0),-127);
    EXPECT_EQ(ix(2,0),0);

    EXPECT_EQ(iy(0,0),127);
    EXPECT_EQ(iy(1,0),0);
    EXPECT_EQ(iy(2,0),0);


    EXPECT_EQ(r(0,0),127);
    EXPECT_EQ(r(1,0),127);
    EXPECT_EQ(r(2,0),0);


}

TEST(AlgorithmTest, Resize)
{
    Eigen::Matrix<std::uint8_t ,4,4> m;
    m << 128,128,128,128,
            128,128,255,255,
            255,128,255,255,
            255,255,255,255;

    const Image mRes = algorithm::resize(m,0.5);
    EXPECT_EQ(mRes(0,0),128U);
    EXPECT_EQ(mRes(1,1),255U);

    EXPECT_EQ(mRes.rows(),2);
    EXPECT_EQ(mRes.cols(),2);



}
