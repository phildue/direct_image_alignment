//
// Created by phil on 30.06.21.
//

#include "Point3D.h"
namespace pd{
    namespace vision {

        Point3D::Point3D(const Eigen::Vector3d& position,std::shared_ptr<const Feature2D> ft)
                : _position(position)
        {
            addFeature(ft);
        }

        void Point3D::addFeature(std::shared_ptr<const Feature2D> ft) {
            _features.push_back(ft);
        }

        void Point3D::removeFeatures() {
            _features.clear();
        }


    }
}