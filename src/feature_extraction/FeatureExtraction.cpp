//
// Created by phil on 07.08.21.
//

#include <core/algorithm.h>


#include "utils/Log.h"
#include "FeatureExtraction.h"
#include "core/Frame.h"
namespace pd{ namespace vision
{

    std::vector<KeyPoint>
    KeyPointExtractorGradientMagnitude::extract(Frame::ShConstPtr frame) const {

        std::vector<KeyPoint> keyPoints;
        keyPoints.reserve(frame->width() * frame->height());
        for (int level = _levels -1 ; level >= 0; --level)
        {
            VLOG(4) << "Extracting features for frame [" << frame->_id << "] at level [" << level <<"]";
            const auto& img = frame->gradientImage(level);

            for (int i = 0; i < frame->height(); i++)
            {
                for (int j = 0; j < frame->width(); j++)
                {
                    if ( img(i,j) >= _threshold )
                    {
                        Eigen::Vector2d pos(j + 0.5 ,i + 0.5);
                        keyPoints.push_back(KeyPoint{pos,std::make_shared<GradientDescriptor>(img(i,j))});
                    }
                }
            }

        }
        return keyPoints;

    }

    KeyPointExtractorGradientMagnitude::KeyPointExtractorGradientMagnitude(int levels, int threshold)
    : _levels( levels )
    , _threshold( threshold )
    {
    }

        FeatureExtractionImpl::FeatureExtractionImpl(int nDesiredFeatures, std::shared_ptr<KeyPointExtractor> kpExtractor )
    : _nDesiredFeatures( nDesiredFeatures )
    , _kpExtractor( kpExtractor )
    {

    }

    void FeatureExtractionImpl::extractFeatures(Frame::ShPtr frame) const
    {
        std::vector<KeyPoint> keyPoints = _kpExtractor->extract(frame);

        if ( !keyPoints.empty() )
        {
            std::partial_sort(keyPoints.begin(), keyPoints.begin() + _nDesiredFeatures, keyPoints.end(), [](auto kp1, auto kp2){
                return (kp2.descriptor->mat().norm() > kp2.descriptor->mat().norm());});
            for ( int i = 0; i < _nDesiredFeatures && i < keyPoints.size(); i++ )
            {
                frame->addFeature(std::make_shared<Feature2D>(keyPoints[i].position,keyPoints[i].descriptor,frame));
            }

        }
        VLOG(3) << "Extracted: ["<< keyPoints.size() << "] features.";

        Log::logFeatures(frame, 3, 4,true, "Features");


    }


    }}