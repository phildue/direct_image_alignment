//
// Created by phil on 07.08.21.
//

#ifndef DIRECT_IMAGE_ALIGNMENT_FEATUREEXTRACTION_H
#define DIRECT_IMAGE_ALIGNMENT_FEATUREEXTRACTION_H

#include <vector>
#include "core/Frame.h"
#include "core/types.h"
#include "Descriptor.h"
namespace pd { namespace vision{

    class KeyPointExtractor;
    class FeatureExtraction
    {
    public:
        virtual void extractFeatures(Frame::ShPtr frame) const = 0;

    protected:
    };

    class FeatureExtractionImpl : public FeatureExtraction
    {
    public:
        explicit FeatureExtractionImpl(int desiredFeatures, std::shared_ptr<KeyPointExtractor> kpExtractor);
        void extractFeatures(Frame::ShPtr frame) const override ;

    protected:
        int _nDesiredFeatures;
        std::shared_ptr<KeyPointExtractor> _kpExtractor;
    };

    //TODO would be nice to be able to stack feature extractors in sort of a pipe / filter pattern
    struct KeyPoint
    {
        Eigen::Vector2d position;
        std::shared_ptr<Descriptor> descriptor;
    };
    class KeyPointExtractor {
    public:
        virtual std::vector<KeyPoint> extract(Frame::ShConstPtr img) const = 0;
    };

    class KeyPointExtractorGradientMagnitude : public KeyPointExtractor
    {
    public:
        KeyPointExtractorGradientMagnitude(int levels, int threshold);
        std::vector<KeyPoint> extract(Frame::ShConstPtr img) const override;

    protected:
        int _threshold;
        int _levels;
    };


} }


#endif //DIRECT_IMAGE_ALIGNMENT_FEATUREEXTRACTION_H
