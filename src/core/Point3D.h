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
            Point3D(const Eigen::Vector3d& position, std::shared_ptr< Feature2D> ft);
            void addFeature(std::shared_ptr<Feature2D> ft);
            void removeFeatures();
            void removeFeature(std::shared_ptr<Feature2D> f);
            void remove();

            const Eigen::Vector3d& position() const { return _position;}
            Eigen::Vector3d position() { return _position;}

            std::vector<std::shared_ptr<Feature2D>> features() {return _features;}
            std::vector<std::shared_ptr<const Feature2D>> features() const {return std::vector<std::shared_ptr<const Feature2D>>(_features.begin(),_features.end());}

            const std::uint64_t _id;
        private:
            std::vector<std::shared_ptr< Feature2D>> _features;
            Eigen::Vector3d _position;
            static std::uint64_t _idCtr;
        };

    }}
#endif //DIRECT_IMAGE_ALIGNMENT_POINT_H
