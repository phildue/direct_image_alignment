//
// Created by phil on 10.10.20.
//

#include <gtest/gtest.h>
#include "core/Frame.h"
#include "utils/Exceptions.h"
using namespace testing;
using namespace pd;
using namespace pd::vision;

TEST(FrameTest,Constructor)
{

    auto camera =std::make_shared<Camera>(10,25,25);
    Eigen::Matrix<std::uint8_t ,50,50> mat;
    auto frame = std::make_shared<Frame>(mat,camera,1);
    EXPECT_EQ(frame->width(),50);
    EXPECT_EQ(frame->height(),50);
    auto img = frame->grayImage(0);
    auto dImg = frame->gradientImage(0);
    EXPECT_EQ(img.cols(), dImg.cols());
    EXPECT_EQ(img.rows(), dImg.rows());

}

TEST(FrameTest,ConstructorDeath)
{
    auto camera =std::make_shared<Camera>(10,25,25);
    Eigen::Matrix<std::uint8_t  ,50,50> mat;
    EXPECT_THROW(std::make_shared<Frame>(mat,camera,0),pd::Exception);
}

TEST(FrameTest,LevelDeath)
{
    auto camera =std::make_shared<Camera>(10,25,25);
    Eigen::Matrix<std::uint8_t  ,50,50> mat;
    auto frame = std::make_shared<Frame>(mat,camera);
    EXPECT_THROW(frame->grayImage(1),pd::Exception);
}

