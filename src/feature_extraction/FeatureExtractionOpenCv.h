//
// Created by phil on 19.08.21.
//

#ifndef VSLAM_FEATUREEXTRACTIONOPENCV_H
#define VSLAM_FEATUREEXTRACTIONOPENCV_H

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "FeatureExtraction.h"
#include "core/Frame.h"
namespace pd{ namespace vision {

    class FeatureExtractionOpenCv : public FeatureExtraction {
    public:

        class FastDescriptor : public Descriptor {
        public:
            FastDescriptor(const Eigen::MatrixXd &descriptor)
                    : _descr(descriptor) {}

            Eigen::MatrixXd mat() const override { return Eigen::Matrix<double, 1, 1>(_descr); };

        private:
            Eigen::VectorXd _descr;
        };

        explicit FeatureExtractionOpenCv(int desiredFeatures, cv::Ptr<cv::Feature2D> detector = cv::FastFeatureDetector::create());

        void extractFeatures(Frame::ShPtr frame) const override;

    private:
        const cv::Ptr<cv::Feature2D> _detector;
        const int _nDesiredFeatures;
    };
}}

#endif //VSLAM_FEATUREEXTRACTIONOPENCV_H
