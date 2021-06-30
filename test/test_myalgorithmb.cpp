//
// Created by phil on 10.10.20.
//

#include <gtest/gtest.h>
#include "MyLibrary/MyLibrary.h"
using namespace testing;
using namespace dnae;
using namespace dnae::MyLibrary;


TEST(MyAlgorithmUnit,Detect)
{
    MyAlgorithm myAlgorithm;

    EXPECT_EQ(false,myAlgorithm.detector(cv::Mat()));
}