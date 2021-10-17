#ifndef DIRECT_IMAGE_IMAGE_ALIGNMENT_SPARSE_H__
#define DIRECT_IMAGE_IMAGE_ALIGNMENT_SPARSE_H__


#include "core/Frame.h"

namespace pd{namespace vision{

    class ImageAlignmentDense
    {
    public:
        explicit ImageAlignmentDense(uint32_t levelMax, uint32_t levelMin, uint32_t stepSizeX = 1, uint32_t stepSizeY = 1);
        void align(FrameRGBD::ShConstPtr referenceFrame, Frame::ShPtr targetFrame) const;

    protected:
        int countValidPoints(FrameRGBD::ShConstPtr referenceFrame, int level) const;
        const int _levelMax,_levelMin;
        const int _stepSizeX, _stepSizeY;

    };



}}

#endif //IMAGE_ALIGNMENT_H__
