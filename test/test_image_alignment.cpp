//
// Created by phil on 10.10.20.
//

#include <gtest/gtest.h>
#include "direct_image_alignment/direct_image_alignment.h"
using namespace testing;
using namespace pd;
using namespace pd::vision;

TEST(MyAlgorithmUnit,Compute)
{
    ImageAlignment imageAlignment(0,0,7);

    auto camera =std::make_shared<Camera>(10,25,25);
    Eigen::Matrix<double,50,50> imgRef,imgTarget;
    auto frameRef = std::make_shared<Frame>(imgRef,camera,0);
    auto frameTarget = std::make_shared<Frame>(imgRef,camera,0);

    auto pose = imageAlignment.align(frameRef,frameTarget);
    //Load reference frame with 3D data
    //Load target frame
    //Align

    EXPECT_TRUE(pose.translation().norm() > 1);
}

