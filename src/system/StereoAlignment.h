//
// Created by phil on 07.08.21.
//

#ifndef VSLAM_STEREO_ALIGNMENT_H
#define VSLAM_STEREO_ALIGNMENT_H

#include "feature_extraction/FeatureExtraction.h"
#include "image_alignment/ImageAlignmentSparse.h"
#include "core/types.h"
#include "sophus/so3.hpp"
namespace pd{ namespace vision{

class StereoAlignment {

public:
    struct Config
    {
        int desiredFeatures;
        int minGradient;
        int levelMax;
        int levelMin;
        int patchSize;
        double cx;
        double cy;
        double fx;
        double fy;
    };

    explicit StereoAlignment(const Config& config);
    Sophus::SE3d align(const Image& img, const Eigen::MatrixXd& depthMap, Timestamp t);

protected:
    const Config _config;
    const std::shared_ptr<const FeatureExtraction> _featureExtractor;
    const std::shared_ptr<const ImageAlignmentSparse> _imageAlignment;
    const std::shared_ptr<const Camera> _camera;
    bool _firstFrame;
    Frame::ShPtr _frameRef;
};

    }}

#endif //SRC_VSLAM_H
