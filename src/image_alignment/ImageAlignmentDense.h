#ifndef DIRECT_IMAGE_IMAGE_ALIGNMENT_SPARSE_H__
#define DIRECT_IMAGE_IMAGE_ALIGNMENT_SPARSE_H__


#include "core/Frame.h"

namespace pd{namespace vision{

    class ImageAlignmentDense
    {
    public:
        explicit ImageAlignmentDense(uint32_t levelMax, uint32_t levelMin);
        void align(FrameRGBD::ShConstPtr referenceFrame, Frame::ShConstPtr targetFrame) const;

    protected:
        const int _levelMax,_levelMin;
    };



}}

#endif //IMAGE_ALIGNMENT_H__
