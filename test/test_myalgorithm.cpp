//
// Created by phil on 10.10.20.
//

#include <gtest/gtest.h>
#include "MyLibrary/MyLibrary.h"
using namespace testing;
using namespace pd;
using namespace pd::MyLibrary;

TEST(MyAlgorithmUnit,Compute)
{
    MyAlgorithm myAlgorithm;
    
    EXPECT_EQ(2,myAlgorithm.compute(1));
}

