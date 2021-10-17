//
// Created by phil on 07.08.21.
//
#include "feature_extraction/feature_extraction.h"
#include "StereoAlignment.h"
#include "core/Point3D.h"
#include "core/Feature2D.h"
#include "core/Frame.h"
#include "core/algorithm.h"
#include "utils/Log.h"
#include "utils/visuals.h"

namespace pd { namespace  vision{

    Sophus::SE3d StereoAlignment::align(const Image &img, const Eigen::MatrixXd &depthMap, Timestamp t) {

        SYSTEM( DEBUG ) << "Align";

        Sophus::SE3d egoMotion;
        auto curFrame = std::make_shared<FrameRGBD>(depthMap, img,_camera,_config.levelMax + 1 );
        Log::getImageLog("System")->append(&vis::drawFrame,std::shared_ptr<const Frame>(curFrame));
        try
        {
            if ( _fNo == 0 )
            {
                SYSTEM( DEBUG ) << "First frame.";
                _frameRef = curFrame;

            }else {

                SYSTEM( DEBUG ) << "New frame..";
                if ( _fNo > 2)
                {
                    curFrame->setPose(_motionPredictor->predict(t));
                }
                _imageAlignment->align(_frameRef,curFrame);

                egoMotion = algorithm::computeRelativeTransform(_frameRef->pose(),curFrame->pose());

                _frameRef = curFrame;
                _motionPredictor->update(_frameRef->pose(),t);

                SYSTEM( DEBUG )  << "Pose: " << curFrame->pose().log().transpose();

            }
                _fNo++;
        }
        catch(const std::exception& e)
        {
            SYSTEM( ERROR ) << e.what();
            _fNo = 0;
        }
        
        return curFrame->pose();
    }

    StereoAlignment::StereoAlignment(const StereoAlignment::Config &config)
    : _config(config)
    , _featureExtractor(std::make_shared<FeatureExtractionOpenCv>(config.desiredFeatures))
    , _imageAlignment(std::make_shared<ImageAlignmentDense>(config.levelMax,config.levelMin))
    , _fNo(0)
    , _camera(std::make_shared<Camera>(config.fx,config.cx,config.cy))
    , _motionPredictor(std::make_shared<MotionPrediction>())
    {
        Log::get("system");
        SYSTEM(INFO) << "Stereoaligner constructed with camera: " << config.fx << " , " << config.cx << " , " << config.cy;
    }
}}