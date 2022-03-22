//
// Created by phil on 10.10.20.
//

#include <gtest/gtest.h>
#include <core/core.h>
#include <utils/utils.h>
#include <opencv2/highgui.hpp>
using namespace testing;
using namespace pd;
using namespace pd::vision;

TEST(FrameTest,CreatePyramid)
{
    auto depth = utils::loadImage(TEST_RESOURCE"/depth.png").cast<double>();
    auto img = utils::loadImage(TEST_RESOURCE"/rgb.png");
    auto cam = std::make_shared<Camera>(1,img.cols()/2,img.rows()/2);
    auto f = std::make_shared<pd::vision::RgbdPyramid>(img,depth,cam,3,0);
    for(size_t i = 0; i < f->nLevels(); i++)
    {
        cv::imshow("Image",vis::drawMat(f->intensity(i)));
        cv::imshow("dIx",vis::drawAsImage(f->dIx(i).cast<double>()));
        cv::imshow("dIy",vis::drawAsImage(f->dIy(i).cast<double>()));
        cv::imshow("Depth",vis::drawAsImage(f->depth(i)));
        cv::waitKey(0);
    }
}
