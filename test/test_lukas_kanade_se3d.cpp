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

class LukasKanadeSE3dTest : public TestWithParam<int>{
    public:
    Image img0,img1;
    Sophus::SE3d pose;
    Eigen::Vector6d x;
    Eigen::MatrixXd depth;
    std::shared_ptr<Camera> camera;
    LukasKanadeSE3dTest()
    {
        img0 = utils::loadImage(TEST_RESOURCE"/sim.png",0,0,true);
        depth = utils::loadDepth(TEST_RESOURCE"/sim.exr");
        
        img0 = algorithm::resize(img0,0.25);
        depth = algorithm::resize(depth,0.25);
        //img0 = utils::loadImage(TEST_RESOURCE"/person.jpg",50,50,true);
        //depth = Eigen::MatrixXd::Ones(img0.rows(),img0.cols())*110;
        camera = std::make_shared<Camera>(381/4,img0.cols()/2,img0.rows()/2);
        img1 = img0;
        x << random::U(0.01,0.011) * random::sign(),random::U(0.01,0.011) * random::sign(),random::U(0.01,0.011) * random::sign(),
        random::U(0.001,0.0011) * random::sign(),random::U(0.001,0.0011) * random::sign(),random::U(0.001,0.0011) * random::sign();
        pose = Sophus::SE3d::exp(x);
   
    }
};

TEST_P(LukasKanadeSE3dTest,LukasKanadeSE3d)
{
    auto mat0 = vis::drawMat(img0);
    auto mat1 = vis::drawMat(img1);

    Log::getImageLog("I")->append(mat0);
    Log::getImageLog("T")->append(mat1);

    auto lk = std::make_shared<LukasKanadeSE3d>(img1,img0,depth,camera,100,1e-5);
    const auto errorInit = (x - Eigen::Vector6d::Zero()).norm();
    ASSERT_FALSE(errorInit <= 0.01);

    lk->solve(x);
    
    std::cout << "Result:\n" << x.transpose() << "\nTrue:\n"<< Eigen::Vector6d::Zero().transpose() << std::endl;
    EXPECT_LT((x - Eigen::Vector6d::Zero()).norm(),0.1*errorInit);
}
INSTANTIATE_TEST_CASE_P(Instantiation, LukasKanadeSE3dTest, ::testing::Range(1, 11));
