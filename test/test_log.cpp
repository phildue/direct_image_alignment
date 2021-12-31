//
// Created by phil on 10.10.20.
//

#include <gtest/gtest.h>
#include "utils/Log.h"
#include "utils/Exceptions.h"
using namespace testing;
using namespace pd;
using namespace pd::vision;
#define VISUALIZE false
TEST(LogTest,Plot)
{

    plt::plot({1,3,2,4});
    if(VISUALIZE)
    {
        plt::show();
    }
}
