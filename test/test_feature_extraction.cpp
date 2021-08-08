//
// Created by phil on 07.08.21.
//

#include <gtest/gtest.h>
#include "feature_extraction/FeatureExtraction.h"
#include "core/Camera.h"
#include "core/Frame.h"

using namespace pd;
using namespace pd::vision;
TEST(FeatureExtractionTest, FeatureExtractionLowThreshold)
{
    auto camera =std::make_shared<Camera>(1,25,25);
    Eigen::Matrix<int,50,50> img = Eigen::Matrix<int, 50, 50>::Ones();

    auto frameRef = std::make_shared<Frame>(img,camera);

    auto featureExtraction = std::make_shared<FeatureExtraction>(5,std::make_shared<KeyPointExtractorGradientMagnitude>(0, 128));
    featureExtraction->extractFeatures(frameRef);
    EXPECT_EQ(frameRef->features().size(), 0);
}

TEST(FeatureExtractionTest, FeatureExtraction)
{
    auto camera =std::make_shared<Camera>(1,25,25);
    Eigen::Matrix<int,50,50> img ;
    img.setRandom();

    auto frameRef = std::make_shared<Frame>(img,camera);

    auto featureExtraction = std::make_shared<FeatureExtraction>(5,std::make_shared<KeyPointExtractorGradientMagnitude>(0, 128));
    featureExtraction->extractFeatures(frameRef);
    EXPECT_EQ(frameRef->features().size(), 5);
}

TEST(FeatureExtractionTest, KeyPointExtraction)
{
    auto camera =std::make_shared<Camera>(1,25,25);
    Eigen::Matrix<int,50,50> img = Eigen::Matrix<int, 50, 50>::Ones();

    auto frameRef = std::make_shared<Frame>(img,camera);
    auto keyPointExtr = std::make_shared<KeyPointExtractorGradientMagnitude>(0, 128);
    auto kps = keyPointExtr->extract(frameRef);
    EXPECT_EQ(kps.size(),0);
}

