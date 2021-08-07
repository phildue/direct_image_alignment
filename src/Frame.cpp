#include <utility>

//
// Created by phil on 30.06.21.
//

#include "Frame.h"
#include "math.h"
#include "Exceptions.h"
namespace pd{
    namespace vision{

        Frame::Frame(const Eigen::MatrixXi &grayImage, Camera::ConstShPtr camera, uint32_t levels,
                     const Sophus::SE3d &pose)
        : _camera(std::move(camera))
        , _pose(pose)
        , _poseInv(pose.inverse())
        {
            if ( levels < 1  )
            {
                throw pd::Exception("Frame needs at least 1 level");
            }
            _grayImagePyramid.resize(levels);
            _grayImagePyramid[0] = grayImage;
            for ( uint32_t i = 1 ; i < levels; i++)
            {
                _grayImagePyramid.push_back(math::resize(grayImage,1.0 / (1U << i)));
            }
        }
        const Eigen::MatrixXi& Frame::grayImage(int level) const {
            if ( level >= levels()  )
            {
                throw pd::Exception("Frame has only [" + std::to_string(levels()) + "]");
            }
            return _grayImagePyramid[level];
        }

        Eigen::MatrixXi& Frame::grayImage(int level) {
            if ( level >= levels()  )
            {
                throw pd::Exception("Frame has only [" + std::to_string(levels()) + "]");
            }
            return _grayImagePyramid[level];
        }

        Eigen::Vector2d Frame::world2image(const Eigen::Vector3d &pWorld) const
        {
            return camera2image(_pose * pWorld);
        }
        Eigen::Vector3d Frame::image2world(const Eigen::Vector2d &pImage, double depth) const
        {
            return _poseInv * image2camera(pImage,depth);

        }
        Eigen::Vector2d Frame::camera2image(const Eigen::Vector3d &pCamera) const
        {
            return camera()->camera2image(pCamera);
        }
        Eigen::Vector3d Frame::image2camera(const Eigen::Vector2d &pImage, double depth) const
        {
            return camera()->image2camera(pImage,depth);
        }

        bool Frame::isVisible(const Eigen::Vector2d &pImage, double border, uint32_t level) const {
            const auto borderScaled = border * (1U << level);

            const auto withinX = ( 0 < std::floor(pImage.x()) + borderScaled && std::ceil(pImage.x()) + borderScaled < width() );
            const auto withinY = ( 0 < std::floor(pImage.y()) + borderScaled && std::ceil(pImage.y()) + borderScaled < height() );

            return withinX && withinY;

        }
        uint32_t Frame::width(uint32_t level) const {
            return grayImage(level).cols();
        }
        uint32_t Frame::height(uint32_t level) const{
            return grayImage(level).rows();

        }

        uint32_t Frame::nObservedPoints() const
        {
            uint32_t nPoints = 0U;
            for (const auto& f : features())
            {
                nPoints = f->point() ? nPoints + 1 : nPoints;
            }
            return nPoints;
        }

        void Frame::addFeature(Feature2D::ShConstPtr ft) {
            _features.push_back(ft);
        }

        void Frame::removeFeatures() {
            _features.clear();
        }

    }
}