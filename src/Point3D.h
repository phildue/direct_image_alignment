//
// Created by phil on 30.06.21.
//

#ifndef DIRECT_IMAGE_ALIGNMENT_POINT3D_H
#define DIRECT_IMAGE_ALIGNMENT_POINT3D_H

#include <memory>
#include <vector>
#include <Eigen/Dense>
#include "Feature2D.h"

namespace pd{
    namespace vision {
class Feature2D;

        class Point3D {
        public:
            using ShPtr = std::shared_ptr<Point3D>;
            using ShConstPtr = std::shared_ptr<Point3D>;
            const Eigen::Vector3d& position() const { return _position;}
            Eigen::Vector3d position() { return _position;}
        private:
            std::vector<Feature2D::ShConstPtr> _features;
            Eigen::Vector3d _position;
        };

    }}
#endif //DIRECT_IMAGE_ALIGNMENT_POINT_H
