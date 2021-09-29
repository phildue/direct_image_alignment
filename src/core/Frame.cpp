#include <utility>

//
// Created by phil on 30.06.21.
//

#include "Frame.h"
#include "Point3D.h"
#include "algorithm.h"
#include "utils/Exceptions.h"
namespace pd{
    namespace vision{

        std::uint64_t Frame::_idCtr = 0U;
        Frame::Frame(const Image& grayImage, Camera::ConstShPtr camera, uint32_t levels,
                     const Sophus::SE3d &pose)
        : _camera(std::move(camera))
        , _pose(pose)
        , _poseInv(pose.inverse())
        , _id(_idCtr++)
        {
            if ( levels < 1  )
            {
                throw pd::Exception("Frame needs at least 1 level");
            }
            //TODO we could think of a sort of lazy evaluation here and only compute the respective image when needed
            _grayImagePyramid.resize(levels);
            _gradientImagePyramid.resize(levels);
            _grayImagePyramid[0] = grayImage;
            _gradientImagePyramid[0] = algorithm::gradient(_grayImagePyramid[0]);
            for ( uint32_t i = 1 ; i < levels; i++)
            {
                const Image imageAtLevel = algorithm::resize(grayImage,1.0 / (1U << i));
                const Image gradientAtLevel = algorithm::gradient(imageAtLevel);
                _grayImagePyramid[i] = imageAtLevel;
                _gradientImagePyramid[i] = gradientAtLevel;
            }

        }
        const Image& Frame::grayImage(int level) const {
            if ( level >= levels()  )
            {
                throw pd::Exception("Frame has only [" + std::to_string(levels()) + "] level.");
            }
            return _grayImagePyramid[level];
        }

        Image& Frame::grayImage(int level) {
            if ( level >= levels()  )
            {
                throw pd::Exception("Frame has only [" + std::to_string(levels()) + "] level.");
            }
            return _grayImagePyramid[level];
        }

        const Image& Frame::gradientImage(int level) const {
            if ( level >= levels()  )
            {
                throw pd::Exception("Frame has only [" + std::to_string(levels()) + "] level.");
            }
            return _gradientImagePyramid[level];
        }

        Image& Frame::gradientImage(int level) {
            if ( level >= levels()  )
            {
                throw pd::Exception("Frame has only [" + std::to_string(levels()) + "] level.");
            }
            return _gradientImagePyramid[level];
        }

        Eigen::Vector2d Frame::world2image(const Eigen::Vector3d &pWorld) const
        {
            return camera2image(world2frame(pWorld));
        }

        Eigen::Vector3d Frame::world2frame(const Eigen::Vector3d &pWorld) const
        {
            return _pose * pWorld;
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

            const auto withinX = ( borderScaled < std::floor(pImage.x()) && std::ceil(pImage.x()) < width() - borderScaled );
            const auto withinY = ( borderScaled < std::floor(pImage.y()) && std::ceil(pImage.y()) < height() - borderScaled );

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

        void Frame::addFeature(Feature2D::ShPtr ft) {
            _features.push_back(ft);
        }

        void Frame::removeFeatures() {

            for (const auto& ft : _features)
            {
                ft->frame() = nullptr;
                if ( ft->point() )
                {
                    ft->point()->removeFeature(ft);
                    ft->point() = nullptr;
                }
            }

            _features.clear();
        }
        void Frame::removeFeature(std::shared_ptr< Feature2D> ft)
        {
            auto it = std::find(_features.begin(),_features.end(),ft);

            if (it == _features.end())
            {
                throw pd::Exception("Did not find feature: [" + std::to_string(ft->id()) + " ] in frame: [" + std::to_string(_id) +"]");
            }
            _features.erase(it);
            ft->frame() = nullptr;

            if ( ft->point() )
            {
                ft->point()->removeFeature(ft);
            }
        }

        Frame::~Frame()
        {
            removeFeatures();
        }

        void Frame::setPose(const Sophus::SE3d &pose) {
            _pose = pose;
            _poseInv = pose.inverse();
        }

        FrameRGBD::FrameRGBD(const Eigen::MatrixXd& depthMap, const Image& grayImage, Camera::ConstShPtr camera,uint32_t levels, const Sophus::SE3d& pose)
        :Frame(grayImage,camera,levels,pose)
        {
            //TODO we could think of a sort of lazy evaluation here and only compute the respective image when needed
            _depthImagePyramid.resize(levels);
            _depthImagePyramid[0] = depthMap;
            for ( uint32_t i = 1 ; i < levels; i++)
            {
                const Eigen::MatrixXd dmAtLevel = algorithm::resize(depthMap,1.0 / (1U << i));
                _depthImagePyramid[i] = dmAtLevel;
            }

        }
        const Eigen::MatrixXd& FrameRGBD::depthMap(int level) const
        {
            if ( level >= levels()  )
            {
                throw pd::Exception("Frame has only [" + std::to_string(levels()) + "] level.");
            }
            return _depthImagePyramid[level];

        }
        Eigen::MatrixXd& FrameRGBD::depthMap(int level)
        {
            if ( level >= levels()  )
            {
                throw pd::Exception("Frame has only [" + std::to_string(levels()) + "] level.");
            }
            return _depthImagePyramid[level];

        }

    }
}