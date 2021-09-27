//
// Created by phil on 10.10.20.
//

#include <gtest/gtest.h>
#include "system/RgbdAlignment.h"
#include "core/Point3D.h"
#include "core/Feature2D.h"
#include "utils/Log.h"
using namespace testing;
using namespace pd;
using namespace pd::vision;

TEST(StereoAlignmentTest,Align)
{
    Log::init(4);
    RgbdAlignment::Config config{};
    RgbdAlignment stereoAlignment(config);

    Eigen::Matrix<std::uint8_t, 50, 50 >img1,img2;
    Eigen::Matrix<double,50,50> depth1,depth2;
    stereoAlignment.align(img1,depth1,0);
    stereoAlignment.align(img2,depth2,1);
}

