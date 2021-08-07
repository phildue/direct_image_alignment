//
// Created by phil on 30.06.21.
//

#ifndef DIRECT_IMAGE_ALIGNMENT_POINT3D_H
#define DIRECT_IMAGE_ALIGNMENT_POINT3D_H

#include <memory>
#include <vector>
#include <Eigen/Dense>

namespace pd{
    namespace vision {
class Feature2D;

        class Point3D {
        public:
            using ShPtr = std::shared_ptr<Point3D>;
            using ShConstPtr = std::shared_ptr<Point3D>;
            Point3D(const Eigen::Vector3d& position, std::shared_ptr<const Feature2D> ft);
            void addFeature(std::shared_ptr<const Feature2D> ft);
            void removeFeatures();

            const Eigen::Vector3d& position() const { return _position;}
            Eigen::Vector3d position() { return _position;}
            const std::vector<std::shared_ptr<const Feature2D>>& features() const {return _features;}
        private:
            std::vector<std::shared_ptr<const Feature2D>> _features;
            Eigen::Vector3d _position;
        };

    }}
#endif //DIRECT_IMAGE_ALIGNMENT_POINT_H
