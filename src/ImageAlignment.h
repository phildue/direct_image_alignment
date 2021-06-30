#ifndef DIRECT_IMAGE_IMAGE_ALIGNMENT_H__
#define DIRECT_IMAGE_IMAGE_ALIGNMENT_H__


#include "Frame.h"
#include "Pose.h"

namespace pd{
namespace vision{
class ImageAlignment
{
public:

    Pose::ShConstPtr align(Frame::ShConstPtr referenceFrame, Frame::ShConstPtr targetFrame);

};

}}
#endif //IMAGE_ALIGNMENT_H__
