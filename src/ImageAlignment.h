#ifndef DIRECT_IMAGE_IMAGE_ALIGNMENT_H__
#define DIRECT_IMAGE_IMAGE_ALIGNMENT_H__


#include "Frame.h"
#include "Pose.h"

namespace pd{
namespace vision{
class ImageAlignment
{
public:
    explicit ImageAlignment(uint32_t levelMax, uint32_t levelMin, uint32_t patchSize);
    Sophus::SE3d align(Frame::ShConstPtr referenceFrame, Frame::ShConstPtr targetFrame);
private:
    const uint32_t _levelMax,_levelMin;
    const uint32_t _patchSize;
};

}}
#endif //IMAGE_ALIGNMENT_H__
