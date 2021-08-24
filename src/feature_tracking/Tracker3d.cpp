//
// Created by phil on 19.08.21.
//

#include "Tracker3d.h"
#include "core/Frame.h"
#include "core/algorithm.h"
#include "core/Point3D.h"
namespace pd{ namespace vision{

    void Tracker3d::track(Frame::ShConstPtr referenceFrame, Frame::ShConstPtr targetFrame) const
     {
         for (const auto& ft : referenceFrame->features() )
         {
             if (ft->point())
             {
                 const auto pTarget = targetFrame->world2image(ft->point()->position());


             }

         }
     }

}}
