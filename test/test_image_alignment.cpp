//
// Created by phil on 10.10.20.
//

#include <gtest/gtest.h>
#include "image_alignment/ImageAlignment.h"
#include "core/Point3D.h"
#include "core/Feature2D.h"
#include "utils/Log.h"
using namespace testing;
using namespace pd;
using namespace pd::vision;


class ImageAlignmentTest : public testing::Test
{
public:
    const double maxErr = 0.1;
    Image imgRef,imgTarget;
    std::shared_ptr<Camera> camera;
    Frame::ShPtr  frameRef,frameTarget;

    ImageAlignmentTest()
    {
        Log::init(5,0,0);

    }

    void createImages(int width, int height, const Sophus::SE3d& pose,double depth, bool random = false)
    {
        camera = std::make_shared<Camera>(1,width/2,height/2);
        imgRef.resize(height,width);
        imgTarget.resize(height,width);

        imgRef << 0,0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,0,
                0,0,0,128,0,0,0,0,0,
                0,0,0,128,0,0,0,0,0,
                0,0,0,128,0,0,0,0,0,
                0,0,0,128,0,0,0,0,0,
                0,0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,0;

        if (random)
        {
            imgRef.setRandom();
        }

        frameRef = std::make_shared<Frame>(imgRef,camera);

        for ( int r = 0; r < imgTarget.rows(); r++)
        {
            for ( int c = 0; c < imgTarget.cols(); c++)
            {
                auto pRef = camera->camera2image( pose.inverse() * camera->image2camera({c,r},depth) ).cast<int>();

                if (0 < pRef.x() && pRef.x() < imgRef.cols() && 0 < pRef.y() && pRef.y() < imgRef.rows())
                {
                    imgTarget(r,c) = algorithm::bilinearInterpolation(imgRef,pRef.x(),pRef.y());

                }else{
                    imgTarget(r,c) = 0;
                }
            }
        }
        frameTarget = std::make_shared<Frame>(imgTarget,camera);
        VLOG(3) << "\n" << imgRef << "\n-->" << "\n" << imgTarget;

    }

    Point3D::ShPtr setupPoint(const Eigen::Vector2d& pImg, double depth)
    {
        auto gradImage = algorithm::gradient(imgRef);
        auto gradMag = algorithm::bilinearInterpolation(gradImage,pImg.x(),pImg.y());
        auto feature = std::make_shared<Feature2D>(pImg,std::make_shared<GradientDescriptor>(gradMag),frameRef);
        frameRef->addFeature(feature);

        auto p3d = frameRef->image2world(pImg,depth);
        feature->point() = std::make_shared<Point3D>(p3d,feature);
        return feature->point();
    }
};

TEST_F(ImageAlignmentTest,Residual)
{
    SE3d motion;
    motion.translation().x() = 20;
    createImages(9,9,motion,10);
    auto p = setupPoint({3,3},10);
    auto error = std::make_shared< ImageAlignment<2>::Cost>(frameRef->features()[0],frameTarget,0);
    frameTarget->setPose(motion);

    auto parameters = frameTarget->pose().log().data();
    Eigen::Matrix<double,4,1> residuals;
    error->Evaluate(&parameters,residuals.data(), nullptr);


    EXPECT_NEAR(residuals.norm(),0,0.01) << "GT Residual Should be 0";
}



TEST_F(ImageAlignmentTest,AnalyticalDiffGt)
{
    SE3d motion;
    motion.translation().x() = 20;
    createImages(9,9,motion,10);
    auto p = setupPoint({3,3},10);
    ImageAlignment<2> imageAlignment(0,0);

    frameTarget->setPose(motion);
    imageAlignment.align(frameRef,frameTarget);

    auto diffT = frameTarget->pose().inverse() * motion;

    EXPECT_NEAR(frameTarget->pose().translation().x(),motion.translation().x(),maxErr) << "Translation error should be smaller.";
}

TEST_F(ImageAlignmentTest,DenseAnalyticalDiffNoise)
{
    SE3d motion;
    motion.translation().x() = 20;
    createImages(9,9,motion,10);
    for (int i = 0; i < 9; i++)
    {
        for (int j = 0; j < 9; j++)
        {
            if (frameRef->isVisible({i,j},2))
            {
                setupPoint({i,j},10);
            }
        }
    }
    ImageAlignment<1> imageAlignment(0,0);

    SE3d motionGt = motion;
    motion.translation().x() += 2.0;
    frameTarget->setPose(motion);

    imageAlignment.align(frameRef,frameTarget);

    auto diffT = frameTarget->pose().inverse() * motion;

    EXPECT_NEAR(frameTarget->pose().translation().x(),motionGt.translation().x(),maxErr) << "Translation error should be smaller.";
}

TEST_F(ImageAlignmentTest,SparseAnalyticalDiffNoise)
{
    SE3d motion;
    motion.translation().x() = 20;
    createImages(9,9,motion,10);
    for (int i = 0; i < 9; i+=3)
    {
        for (int j = 0; j < 9; j+=3)
        {
            if (frameRef->isVisible({i,j},2))
            {
                setupPoint({i,j},10);
            }
        }
    }
    ImageAlignment<3> imageAlignment(0,0);

    SE3d motionGt = motion;
    motion.translation().x() += 15;
    frameTarget->setPose(motion);

    imageAlignment.align(frameRef,frameTarget);

    auto diffT = frameTarget->pose().inverse() * motion;

    EXPECT_NEAR(frameTarget->pose().translation().x(),motionGt.translation().x(),maxErr) << "Translation error should be smaller.";
}

TEST_F(ImageAlignmentTest,DISABLED_AlignAutoDiffGt)
{
    SE3d motion;
    motion.translation().x() = 20;
    createImages(9,9,motion,10);
    auto p = setupPoint({3,3},10);
    ImageAlignmentAutoDiff<2> imageAlignment(0,0);

    frameTarget->setPose(motion);
    imageAlignment.align(frameRef,frameTarget);

    auto diffT = frameTarget->pose().inverse() * motion;

    EXPECT_NEAR(frameTarget->pose().translation().x(),motion.translation().x(),maxErr) << "Translation error should be smaller.";
}

TEST_F(ImageAlignmentTest,DISABLED_AutoDiffNoise)
{
    SE3d motion;
    motion.translation().x() = 20;
    createImages(9,9,motion,10);
    auto p = setupPoint({3,3},10);
    ImageAlignmentAutoDiff<2> imageAlignment(0,0);

    motion.translation().x() += 0.1;
    frameTarget->setPose(motion);

    imageAlignment.align(frameRef,frameTarget);

    auto diffT = frameTarget->pose().inverse() * motion;

    EXPECT_NEAR(frameTarget->pose().translation().x(),motion.translation().x(),maxErr) << "Translation error should be smaller.";
}

TEST_F(ImageAlignmentTest,DISABLED_DenseAutoDiffNoise)
{
    SE3d motion;
    motion.translation().x() = 20;
    createImages(9,9,motion,10);
    for (int i = 0; i < 9; i++)
    {
        for (int j = 0; j < 9; j++)
        {
            if (frameRef->isVisible({i,j},2))
            {
                setupPoint({i,j},10);
            }
        }
    }
    ImageAlignmentAutoDiff<1> imageAlignment(0,0);

    motion.translation().x() += 15;
    frameTarget->setPose(motion);

    imageAlignment.align(frameRef,frameTarget);

    auto diffT = frameTarget->pose().inverse() * motion;

    EXPECT_NEAR(frameTarget->pose().translation().x(),motion.translation().x(),maxErr) << "Translation error should be smaller.";
}