//
// Created by phil on 19.08.21.
//

#ifndef VSLAM_TRACKER3D_H
#define VSLAM_TRACKER3D_H

#include "core/Frame.h"
namespace pd{ namespace vision{
    class Tracker3d {
    public:
        void track(Frame::ShConstPtr referenceFrame, Frame::ShConstPtr targetFrame) const;

    };
}}


#endif //VSLAM_TRACKER3D_H
