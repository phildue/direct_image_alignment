//
// Created by phil on 19.08.21.
//

#include <memory>

#include <Eigen/Dense>
#include <opencv4/opencv2/core/eigen.hpp>

#include "FeatureExtractionOpenCv.h"
#include "core/Frame.h"
#include "core/Feature2D.h"
#include "utils/utils.h"

namespace pd{ namespace vision{

        FeatureExtractionOpenCv::FeatureExtractionOpenCv(int desiredFeatures, cv::Ptr<cv::Feature2D> detector)
        : _detector(detector)
        , _nDesiredFeatures(desiredFeatures)
        {}

        void FeatureExtractionOpenCv::extractFeatures(Frame::ShPtr frame) const {

            std::vector<cv::KeyPoint> keypoints_fast;
            cv::Ptr<cv::FastFeatureDetector> fast = cv::FastFeatureDetector::create();
            cv::Mat img;
            cv::eigen2cv(frame->grayImage(0),img);
            fast->detect(img, keypoints_fast);
            VLOG(3) << "Found: ["<< keypoints_fast.size() << "] keypoints.";

            std::partial_sort(keypoints_fast.begin(), keypoints_fast.begin() + _nDesiredFeatures, keypoints_fast.end(), [](auto kp1, auto kp2){
                return (kp1.response > kp2.response);});

            for ( int i = 0; i < _nDesiredFeatures && i < keypoints_fast.size(); i++ )
            {
                const auto& kp = keypoints_fast[i];
                auto descr = std::make_shared<FastDescriptor>(Eigen::Matrix<double,1,1>(kp.response));
                frame->addFeature(std::make_shared<Feature2D>(Eigen::Vector2d(kp.pt.x,kp.pt.y),descr,frame, nullptr,kp.octave));
            }

            VLOG(3) << "Created: ["<< frame->features().size() << "] features.";

            Log::logFeatures(frame, 3, 4,false, "Features");

        }
}}