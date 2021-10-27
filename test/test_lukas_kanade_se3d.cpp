//
// Created by phil on 10.10.20.
//

#include <gtest/gtest.h>
#include "utils/Exceptions.h"
#include "utils/Log.h"
#include "utils/utils.h"
#include "utils/visuals.h"
#include "core/algorithm.h"
#include "core/types.h"
#include "lukas_kanade/LukasKanadeSE3d.h"

using namespace testing;
using namespace pd;
using namespace pd::vision;

TEST(LukasKanadeTest,LukasKanadeSE3d)
{
    /*
    Image img0 = utils::loadImage(TEST_RESOURCE"/sim.png",0,0,true);
    Eigen::MatrixXd depth = utils::loadDepth(TEST_RESOURCE"/sim.exr");
    img0 = algorithm::resize(img0,0.25);
    depth = algorithm::resize(depth,0.25);*/
    const Image img0 = utils::loadImage(TEST_RESOURCE"/stuff.png",50,50,true);
    const Eigen::MatrixXd depth = Eigen::MatrixXd::Ones(img0.rows(),img0.cols())*0;
    auto camera = std::make_shared<Camera>(25,img0.cols()/2,img0.rows()/2);
    
    Image img1 = img0;
    auto mat0 = vis::drawMat(img0);
    auto mat1 = vis::drawMat(img1);

    Log::getImageLog("I")->append(mat0);
    Log::getImageLog("T")->append(mat1);

    auto lk = std::make_shared<LukasKanadeSE3d>(img1,img0,depth,camera,50,1e-5);
   
    Sophus::SE3d pose;
    pose.translation().x() = 0.1;
    Eigen::Vector6d x = pose.log();
    ASSERT_FALSE((x - Eigen::Vector6d::Zero()).norm() <= 0.05);

    lk->solve(x);
    
    std::cout << "Result:\n" << x.transpose() << "\nTrue:\n"<< Eigen::Vector6d::Zero().transpose() << std::endl;
    EXPECT_NEAR((x - Eigen::Vector6d::Zero()).norm(),0,0.05);
}
