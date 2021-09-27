//
// Created by phil on 10.10.20.
//

#include <gtest/gtest.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include "image_alignment/ImageAlignment.h"
#include "core/Point3D.h"
#include "core/Feature2D.h"
#include "core/Frame.h"
#include "utils/Log.h"
using namespace testing;
using namespace pd;
using namespace pd::vision;


class ImageAlignmentTest : public testing::Test
{
public:
    const double maxErr = 0.05;
    Image imgRef,imgTarget;
    std::shared_ptr<Camera> camera;
    Frame::ShPtr  frameRef,frameTarget;

    ImageAlignmentTest()
    {
        Log::init(5,0,0);

    }
    void loadRefImage(const std::string &path, int height = -1, int width = -1)
    {
        imgRef = utils::loadImage(path,height,width,true);
        camera = std::make_shared<Camera>(381/4,imgRef.cols()/2,imgRef.rows()/2);
        frameRef = std::make_shared<Frame>(imgRef,camera);
        VLOG(4) << "\n" << imgRef.cast<int>();
    }
    void createTargetImage(const Sophus::SE3d &pose, double depth)
    {
        Eigen::MatrixXd depthMat(imgRef.rows(),imgRef.cols());
        depthMat.setOnes();
        depthMat *= depth;
        createTargetImage(pose,depthMat);
    }

    void createTargetImage(const Sophus::SE3d &pose, const Eigen::MatrixXd& depth)
    {
        imgTarget.resize(imgRef.rows(),imgRef.cols());
        for ( int r = 0; r < imgTarget.rows(); r++)
        {
            for ( int c = 0; c < imgTarget.cols(); c++)
            {
                auto pRef = camera->camera2image( pose.inverse() * camera->image2camera({c,r},depth(r,c)) ).cast<int>();

                if (0 < pRef.x() && pRef.x() < imgRef.cols() && 0 < pRef.y() && pRef.y() < imgRef.rows())
                {
                    imgTarget(r,c) = algorithm::bilinearInterpolation(imgRef,pRef.x(),pRef.y());

                }else{
                    imgTarget(r,c) = 0;
                }
            }
        }
        frameTarget = std::make_shared<Frame>(imgTarget,camera);
        VLOG(4) << "\n-->" << "\n" << imgTarget.cast<int>();

    }

    void createImages(int width, int height, const Sophus::SE3d& pose,double depth, bool random = false)
    {
        camera = std::make_shared<Camera>(1,width/2,height/2);
        imgRef.resize(height,width);
        imgTarget.resize(height,width);


        if (random)
        {
            imgRef.setRandom();
        }else{
            imgRef << 0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,
                    0,0,0,128,0,0,0,0,0,
                    0,0,128,128,0,0,0,0,0,
                    0,0,0,128,128,0,0,0,0,
                    0,0,0,128,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0;

        }
        frameRef = std::make_shared<Frame>(imgRef,camera);

        createTargetImage(pose,depth);

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

TEST_F(ImageAlignmentTest,ResidualZero)
{
    SE3d motion;
    motion.translation().x() = 20;
    createImages(9,9,motion,10);
    frameTarget->setPose(motion);

    auto p = setupPoint({3,3},10);
    auto error = std::make_shared< ImageAlignmentCeres<2>::Cost>(frameRef->features()[0],frameTarget,0);

    auto parameters = frameTarget->pose().log().data();
    Eigen::Matrix<double,4,1> residuals;
    error->Evaluate(&parameters,residuals.data(), nullptr);


    EXPECT_NEAR(residuals.norm(),0,0.01) << "GT Residual Should be 0";
}

TEST_F(ImageAlignmentTest, DISABLED_ResidualNotZero)
{
    SE3d motion;
    motion.translation().x() = 20;
    createImages(9,9,motion,10);
    motion.translation().x() += 10;
    frameTarget->setPose(motion);

    auto p = setupPoint({3,3},10);
    auto error = std::make_shared< ImageAlignmentCeres<2>::Cost>(frameRef->features()[0],frameTarget,0);

    auto parameters = frameTarget->pose().log().data();
    Eigen::Matrix<double,4,1> residuals;
    error->Evaluate(&parameters,residuals.data(), nullptr);


    EXPECT_GT(residuals.norm(),0.1) << "Noisy Residual Should be 0";
}


TEST_F(ImageAlignmentTest,DenseResidual)
{
    Log::init(3,0,0);
    SE3d motion;
    loadRefImage(fs::path(TEST_RESOURCE"/sim.png"),120,160);

    auto depthMat = utils::loadDepth(TEST_RESOURCE"/sim.exr",120,160);

    VLOG(3) << depthMat;

    createTargetImage(motion,depthMat);

    motion.translation().x() += 0.1;
    frameTarget->setPose(motion);

    Eigen::MatrixXd residualImage(imgRef.rows(),imgRef.cols());
    Eigen::MatrixXd jacobianImage(imgRef.rows(),imgRef.cols());
    residualImage.array() = 0;
    jacobianImage.array() = 0;
    double max = 0;
    double min =  std::numeric_limits<double>::max();
    double maxJ = 0;
    double minJ = std::numeric_limits<double>::max();
    EXPECT_FALSE(std::isnan(residualImage.norm()));

    for (int i = 0; i < imgRef.rows(); i++)
    {
        for (int j = 0; j < imgRef.cols(); j++)
        {
            if (frameRef->isVisible({j,i},2))
            {
                auto p = setupPoint({j,i},depthMat(i,j));
                auto error = std::make_shared< ImageAlignmentCeres<1>::Cost>(p->features()[0],frameTarget,0);

                auto parameters = frameTarget->pose().log().data();
                Eigen::Matrix<double,1,1> residuals;
                Eigen::Matrix<double,6,1> jacobians;
                auto J = jacobians.data();
                error->Evaluate(&parameters,residuals.data(), &J);
                const auto resNorm = residuals.norm();
                EXPECT_FALSE(std::isnan(resNorm));
                residualImage(i,j) = resNorm;
                //auto Jnorm = frameTarget->camera()->J_xyz2uv(p->position()).norm();
                auto Jnorm = jacobians[0];
                jacobianImage(i,j) = Jnorm;

                if ( resNorm > max )
                {
                    max = resNorm;
                }
                if ( resNorm < min)
                {
                    min = resNorm;
                }

                if ( Jnorm > maxJ )
                {
                    maxJ = Jnorm;
                }
                if ( Jnorm < minJ)
                {
                    minJ = Jnorm;
                }
            }
        }
    }
    EXPECT_FALSE(std::isnan(residualImage.norm()));
    residualImage.array() -= min;
    residualImage /= (max - min);
    residualImage *= 255;

    jacobianImage.array() -= minJ;
    jacobianImage /= (maxJ - minJ);
    jacobianImage *= 255;

    EXPECT_FALSE(std::isnan(residualImage.norm()));

    VLOG(3) << "max: " << max << " min: " << min << "\n" << residualImage;
    Log::logMat(residualImage.cast<std::uint8_t >(),3,"Residual");
    Log::logMat(jacobianImage.cast<std::uint8_t >(),3,"Jacobian");
    Log::logMat(imgRef,3,"ImgRef");
    Log::logMat(imgTarget,3,"ImgTarget");

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
    Log::init(4,0,0);
    SE3d motion;
    loadRefImage(fs::path(TEST_RESOURCE"/sim.png"),120,160);

    auto depthMat = utils::loadDepth(TEST_RESOURCE"/sim.exr",120,160);

    createTargetImage(motion,depthMat);
    auto motionGt = motion;
    motion.translation().x() += 0.1;
    frameTarget->setPose(motion);

    for (int i = 0; i < imgRef.rows(); i++)
    {
        for (int j = 0; j < imgRef.cols(); j++)
        {
            if (frameRef->isVisible({j,i},2))
            {
                setupPoint({j,i},depthMat(i,j));
            }
        }
    }
    ImageAlignment<1> imageAlignment(0,0);


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

TEST_F(ImageAlignmentTest,DISABLED_DenseAutoDiffNoiseSim)
{
    Log::init(3,0,0);
    SE3d motion;
    loadRefImage(fs::path(TEST_RESOURCE"/sim.png"),120,160);

    auto depthMat = utils::loadDepth(TEST_RESOURCE"/sim.exr",120,160);

    createTargetImage(motion,depthMat);
    auto motionGt = motion;
    motion.translation().x() += 1;
    frameTarget->setPose(motion);

    for (int i = 0; i < imgRef.rows(); i++)
    {
        for (int j = 0; j < imgRef.cols(); j++)
        {
            if (frameRef->isVisible({j,i},2))
            {
                setupPoint({j,i},depthMat(i,j));
            }
        }
    }
    ImageAlignmentAutoDiff<1> imageAlignment(0,0);


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