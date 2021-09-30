//
// Created by phil on 10.10.20.
//

#include <gtest/gtest.h>
#include "utils/Exceptions.h"
#include "utils/Log.h"
#include "motion_prediction/MotionPrediction.h"
using namespace testing;
using namespace pd;
using namespace pd::vision;

TEST(MathTest,BilinearInterpolation)
{
    MotionPrediction predictor;
    for (int t = 1; t < 5; t++)
    {
        if ( t == 1)
        {
            EXPECT_ANY_THROW(predictor.predict(t+1));
        }
        Sophus::SE3d pose;
        pose.translation().x() = t;
        predictor.update(pose,t);
        if ( t == 1)
        {
            EXPECT_ANY_THROW(predictor.predict(t+1));
        }else{
            const auto predictedPose = predictor.predict(t+1);
            EXPECT_EQ(predictedPose.translation().x(), t + 1);
        }


    }
}

