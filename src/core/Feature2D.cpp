//
// Created by phil on 30.06.21.
//

#include "Feature2D.h"
#include "Frame.h"
#include "algorithm.h"

namespace pd{
    namespace vision{

        std::uint64_t Feature2D::_idCtr = 0U;

        Feature2D::Feature2D(const Eigen::Vector2d& position,std::shared_ptr<Descriptor> descriptor, std::shared_ptr<Frame> frame,std::shared_ptr<Point3D> p3d, int level)
        : _position(position)
        , _frame(frame)
        , _point(p3d)
        , _descriptor(descriptor)
        , _id(_idCtr++)
        , _level(level)
        {

        }



    }
}
