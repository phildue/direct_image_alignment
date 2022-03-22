//
// Created by phil on 10.10.20.
//

#include <gtest/gtest.h>
#include <opencv2/highgui.hpp>
#include <core/core.h>
#include <utils/utils.h>
#include "SE3Alignment.h"

using namespace testing;
using namespace pd;
using namespace pd::vision;
using namespace pd::vslam::solver;
TEST(OdometryRgbd,TestOnSyntheticData)
{
        
    auto solver = std::make_shared<GaussNewton<6>>(1e-7,100);
    auto loss = std::make_shared<QuadraticLoss>();
    auto scaler = std::make_shared<Scaler>();
    LOG_IMG("ImageWarped")->_show = true;
    LOG_IMG("Depth")->_show = true;
    LOG_IMG("Residual")->_show = true;
    LOG_IMG("Image")->_show = true;
    LOG_IMG("Depth")->_show = true;
    LOG_IMG("Weights")->_show = true;
    LOG_IMG("Residual")->_block = true;

    auto aligner = std::make_shared<SE3Alignment>(1,solver,loss,scaler);
    
    // tum depth format: https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
    DepthMap depth = utils::loadDepth(TEST_RESOURCE"/depth.png")/5000.0;
    Image img = utils::loadImage(TEST_RESOURCE"/rgb.png");
    auto cam = std::make_shared<Camera>(525.0,525.0,319.5,239.5);
    

    SE3d initialPose(transforms::euler2quaternion(0.06,0.07,0.06),{0.03,0.05,0.03});
    //SE3d initialPose(transforms::euler2quaternion(0.03,0.03,0.03),{0.03,0.05,0.03});
    auto fRef = std::make_shared<RgbdPyramid>(img,depth,cam,3,0);
    auto fCur = std::make_shared<RgbdPyramid>(img,depth,cam,3,1,PoseWithCovariance(initialPose,MatXd::Identity(6,6)));
  
    auto result = aligner->align(fRef,fCur)->pose().log();
    auto angleAxis = result.tail(3);
    const double eps = 0.01;
    EXPECT_NEAR(result.x(),0.0,eps);
    EXPECT_NEAR(result.y(),0.0,eps);
    EXPECT_NEAR(result.z(),0.0,eps);
    EXPECT_NEAR(angleAxis.norm(),0.0,eps);

}

TEST(OdometryRgbd,DISABLED_TestOnSyntheticDataMultiFrame)
{
        
    auto solver = std::make_shared<GaussNewton<6>>(1e-7,100);
    auto loss = std::make_shared<QuadraticLoss>();
    auto scaler = std::make_shared<Scaler>();
    LOG_IMG("ImageWarped")->_show = true;
    LOG_IMG("Depth")->_show = true;
    LOG_IMG("Residual")->_show = true;
    LOG_IMG("Image")->_show = true;
    LOG_IMG("Depth")->_show = true;
    LOG_IMG("Weights")->_show = true;
    LOG_IMG("Residual")->_block = true;

    auto aligner = std::make_shared<SE3Alignment>(1,solver,loss,scaler);
    
    // tum depth format: https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
    DepthMap depth = utils::loadDepth(TEST_RESOURCE"/depth.png")/5000.0;
    Image img = utils::loadImage(TEST_RESOURCE"/rgb.png");
    auto cam = std::make_shared<Camera>(525.0,525.0,319.5,239.5);
    

    //SE3d initialPose(transforms::euler2quaternion(0.06,0.07,0.06),{0.03,0.05,0.03});
    SE3d initialPose(transforms::euler2quaternion(0.03,0.03,0.03),{0.03,0.05,0.03});
    auto fRef0 = std::make_shared<pd::vision::RgbdPyramid>(img,depth,cam,3,0);
    auto fRef1 = std::make_shared<pd::vision::RgbdPyramid>(img,depth,cam,3,1,PoseWithCovariance(initialPose,MatXd::Identity(6,6)));
    auto fCur = std::make_shared<pd::vision::RgbdPyramid>(img,depth,cam,3,1,PoseWithCovariance(initialPose,MatXd::Identity(6,6)));
  
    auto result = aligner->align({fRef0,fRef1},fCur)->pose().log();
    auto angleAxis = result.tail(3);
    const double eps = 0.01;
    EXPECT_NEAR(result.x(),0.0,eps);
    EXPECT_NEAR(result.y(),0.0,eps);
    EXPECT_NEAR(result.z(),0.0,eps);
    EXPECT_NEAR(angleAxis.norm(),0.0,eps);

}