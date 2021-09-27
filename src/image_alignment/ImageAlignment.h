#ifndef DIRECT_IMAGE_IMAGE_ALIGNMENT_H__
#define DIRECT_IMAGE_IMAGE_ALIGNMENT_H__


#include <sophus/se3.hpp>
#include <ceres/ceres.h>

#include "core/Frame.h"

namespace pd{namespace vision{

    template<int patchSize>
    class ImageAlignment
    {
    public:
        explicit ImageAlignment(uint32_t levelMax, uint32_t levelMin);
        virtual void align(Frame::ShConstPtr referenceFrame, Frame::ShConstPtr targetFrame) const;

    protected:
        const int _levelMax,_levelMin;
    };



}}

#include "ImageAlignment.hpp"
#include "ImageAlignmentAutoDiff.hpp"
#include "ImageAlignmentCeres.hpp"
#endif //IMAGE_ALIGNMENT_H__
