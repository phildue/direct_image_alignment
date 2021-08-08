//
// Created by phil on 07.08.21.
//

#include "FeatureExtraction.h"
namespace pd{ namespace vision
{

    std::vector<KeyPoint>
    KeyPointExtractorGradientMagnitude::extract(Frame::ShConstPtr frame) const {

        std::vector<KeyPoint> keyPoints;
        keyPoints.reserve(frame->width() * frame->height());
        for (int level = _levels; level >= 0; --level)
        {
            const auto img = frame->gradientImage(level);

            for (int i = 0; i < frame->height(); i++)
            {
                for (int j = 0; j < frame->width(); j++)
                {
                    if ( img(i,j) > _threshold )
                    {
                        Eigen::Vector2d pos(i,j);
                        keyPoints.push_back(KeyPoint{pos,img(i,j)});
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

    FeatureExtraction::FeatureExtraction(int nDesiredFeatures, std::shared_ptr<KeyPointExtractor> kpExtractor )
    : _nDesiredFeatures( nDesiredFeatures )
    , _kpExtractor( kpExtractor )
    {

    }

    void FeatureExtraction::extractFeatures(Frame::ShPtr frame) const
    {
        std::vector<KeyPoint> keyPoints = _kpExtractor->extract(frame);
        std::partial_sort(keyPoints.begin(), keyPoints.begin() + _nDesiredFeatures, keyPoints.end(), [](auto kp1, auto kp2){ return kp1.value < kp2.value;});
        for ( int i = 0; i < _nDesiredFeatures && i < keyPoints.size(); i++ )
        {
            frame->addFeature(std::make_shared<Feature2D>(keyPoints[i].position,frame));
        }
    }
}}