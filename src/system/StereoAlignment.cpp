//
// Created by phil on 07.08.21.
//

#include "StereoAlignment.h"
namespace pd { namespace  vision{

    Sophus::SE3d StereoAlignment::align(const Image &img, const Eigen::MatrixXd &depthMap, Timestamp t) {

        Sophus::SE3d relativePose;
        auto curFrame = std::make_shared<Frame>(img,_camera,_config.levelMax + 1);

        if ( _firstFrame )
        {
            _frameRef = curFrame;

        }else {
            relativePose = _imageAlignment->align(_frameRef,curFrame);
            _frameRef = curFrame;
            _firstFrame = false;

        }
        _featureExtractor->extractFeatures(_frameRef);
        //TODO for each feature where depth is available create 3d point

        return relativePose;
    }

    StereoAlignment::StereoAlignment(const StereoAlignment::Config &config)
    : _config(config)
    , _featureExtractor(std::make_shared<FeatureExtraction>(config.desiredFeatures,std::make_shared<KeyPointExtractorGradientMagnitude>(config.levelMax,config.minGradient)))
    , _imageAlignment(std::make_shared<ImageAlignment>(config.levelMin,config.levelMax,config.patchSize))
    , _firstFrame(false)
    {

    }
}}