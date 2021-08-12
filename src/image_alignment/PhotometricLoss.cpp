//
// Created by phil on 10.08.21.
//

#include "PhotometricLoss.h"
#include "utils/Exceptions.h"
#include "utils/utils.h"
#include "core/algorithm.h"
#include "core/Point3D.h"
#include "core/Frame.h"
#include "core/Feature2D.h"
namespace pd{ namespace vision{
        PhotometricLoss::PhotometricLoss (Feature2D::ShConstPtr ftRef,
                         Frame::ShConstPtr frameTarget,
                         uint32_t patchSize,
                         uint32_t level)
        : _ftRef(ftRef)
        , _frameTarget(frameTarget)
        , _p3d(ftRef->point()->position())
        , _patchSize(patchSize)
        , _patchArea(patchSize*patchSize)
        , _level(level)
        , _scale(1.0/(1U<<level))
        , _patchSizeHalf(patchSize/2){

            if(!ftRef->point())
            {
                throw pd::Exception("Feature [" + std::to_string(ftRef->id()) + "] does not have corresponding 3D point.");
            }
            if(!ftRef->frame()->isVisible(ftRef->position(),_patchSize,level))
            {
                throw pd::Exception("Feature [" + std::to_string(ftRef->id()) + "] is not fully visible in reference frame.");

            }

            const auto frameRef = ftRef->frame();
            const auto mat = frameRef->grayImage(level);
            const auto pCamera = frameRef->pose() * _p3d;
            const auto jacobianFeature = frameRef->camera()->J_xyz2uv(pCamera);

            utils::throw_if_nan(jacobianFeature,"Point Jac.",ftRef);

            const auto pImg = _ftRef->position();
            _jacobian.resize(6,patchSize*patchSize);
            _patchRef.resize (_patchSize,_patchSize);
            int idxPixel = 0;
            for (int i = 0; i < _patchSize; i++)
            {
                for (int j = 0; j < _patchSize; j++)
                {
                    _patchRef(i,j) = algorithm::bilinearInterpolation(mat,
                                                                      (pImg.y() - _patchSizeHalf + i) *_scale,
                                                                      (pImg.x() - _patchSizeHalf + j) *_scale);

                    const double fx1 = algorithm::bilinearInterpolation(mat,
                                                                        (pImg.y() - _patchSizeHalf + i) *_scale,
                                                                        (pImg.x() - _patchSizeHalf + j + 1) *_scale);
                    const double fx0 = algorithm::bilinearInterpolation(mat,
                                                                        (pImg.y() - _patchSizeHalf + i ) *_scale,
                                                                        (pImg.x() - _patchSizeHalf + j ) *_scale);
                    const double fy1 = algorithm::bilinearInterpolation(mat,
                                                                        (pImg.y() - _patchSizeHalf + i + 1) *_scale,
                                                                        (pImg.x() - _patchSizeHalf + j) *_scale);
                    const double fy0 = algorithm::bilinearInterpolation(mat,
                                                                        (pImg.y() - _patchSizeHalf + i ) *_scale,
                                                                        (pImg.x() - _patchSizeHalf + j ) *_scale);

                    double dx = 0.5 * (fx1 - fx0);
                    double dy = 0.5 * (fy1 - fy0);

                    _jacobian.col(idxPixel++) = (dx*jacobianFeature.row(0) + dy*jacobianFeature.row(1))*(frameRef->camera()->focalLength() / _scale);

                }
            }


            utils::throw_if_nan(_jacobian,"Jac.",ftRef);

            VLOG(5) << "Photometric Loss for feature [ " << ftRef->id() << " ]:"
                    << " \n W = " << _patchRef
                    << " \n J = " << _jacobian;
            this->set_num_residuals(_patchArea);

        }

        bool PhotometricLoss::Evaluate(double const* const* parameters,double* residuals,double** jacobians) const {

            Sophus::SE3d pose;
            const auto pCamera = pose * _p3d;
            const auto pImg = _frameTarget->camera2image(pCamera);

            if ( _frameTarget->isVisible(pImg,_patchSizeHalf,_level))
            {
                int idxPixel = 0;
                for (int i = 0; i < _patchSize; i++)
                {
                    for (int j = 0; j < _patchSize; j++)
                    {
                        auto idxRow = static_cast<std::uint32_t > ( i * _scale);
                        auto idxCol = static_cast<std::uint32_t > ( j * _scale);
                        double grayTarget =     algorithm::bilinearInterpolation(_frameTarget->grayImage(_level),
                                                                                 (pImg.y() - _patchSizeHalf + i) *_scale,
                                                                                 (pImg.x() - _patchSizeHalf + j) *_scale);
                        residuals[idxPixel] = (_patchRef(i,j) - grayTarget);

                        idxPixel++;
                    }

                }
                jacobians[0] = (double*)_jacobian.data();
                return true;
            }else{
                return false;

            }
        }


    }}