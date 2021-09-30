//
// Created by phil on 10.10.20.
//

#include <gtest/gtest.h>
#include "feature_extraction/FeatureExtractionOpenCv.h"
#include "image_alignment/ImageAlignmentDense.h"
#include "core/Point3D.h"
#include "core/Feature2D.h"
#include "core/Frame.h"
#include "utils/Log.h"
#include "utils/utils.h"
#include "core/algorithm.h"

using namespace testing;
using namespace pd;
using namespace pd::vision;


class ImageAlignmentDenseTest : public testing::Test
{
public:
    const double maxErr = 0.05;
    Image imgRef,imgTarget;
    std::shared_ptr<Camera> camera;
    FrameRGBD::ShPtr frameRef;
    Frame::ShPtr frameTarget;

    ImageAlignmentDenseTest()
    {
        Log::init(5,0,0);

    }
    void loadRefImage(const std::string &path,const std::string& pathDepth, int height = -1, int width = -1)
    {
        imgRef = utils::loadImage(path,height,width,true);
        auto depthMat = utils::loadDepth(pathDepth,120,160);
        camera = std::make_shared<Camera>(381/4,imgRef.cols()/2,imgRef.rows()/2);
        frameRef = std::make_shared<FrameRGBD>(depthMat,imgRef,camera);
        VLOG(4) << "\n" << imgRef.cast<int>();
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




};

TEST_F(ImageAlignmentDenseTest,Align)
{
    Log::init(4,0,0);
    SE3d motion;
    loadRefImage(fs::path(TEST_RESOURCE"/sim.png"),fs::path(TEST_RESOURCE"/sim.png"),120,160);

    createTargetImage(motion,frameRef->depthMap(0));

    auto motionGt = motion;
    motion.translation().x() += 0.1;
    frameTarget->setPose(motion);

    ImageAlignmentDense imageAlignment(0,0);


    imageAlignment.align(frameRef,frameTarget);

    auto diffT = frameTarget->pose().inverse() * motion;

    EXPECT_NEAR(frameTarget->pose().translation().x(),motionGt.translation().x(),maxErr) << "Translation error should be smaller.";
}

TEST_F(ImageAlignmentDenseTest,Align2Times)
{
    Log::init(4,0,0);
    SE3d motion;
    loadRefImage(fs::path(TEST_RESOURCE"/sim.png"),fs::path(TEST_RESOURCE"/sim.png"),120,160);

    createTargetImage(motion,frameRef->depthMap(0));

    auto motionGt = motion;
    motion.translation().x() += 0.1;
    frameTarget->setPose(motion);

    ImageAlignmentDense imageAlignment(0,0);


    imageAlignment.align(frameRef,frameTarget);
    imageAlignment.align(frameRef,frameTarget);

    auto diffT = frameTarget->pose().inverse() * motion;

    EXPECT_NEAR(frameTarget->pose().translation().x(),motionGt.translation().x(),maxErr) << "Translation error should be smaller.";
}
