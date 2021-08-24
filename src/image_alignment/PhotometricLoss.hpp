//
// Created by phil on 10.08.21.
//
#include <algorithm>    // std::max

#include "utils/Exceptions.h"
#include "utils/utils.h"
#include "core/algorithm.h"
#include "core/Point3D.h"
#include "core/Frame.h"
#include "core/Feature2D.h"
#include "ImageAlignment.h"
namespace pd{ namespace vision{



        template <int patchSize>
        ImageAlignment<patchSize>::Cost::Cost (Feature2D::ShConstPtr ftRef,
                         Frame::ShConstPtr frameTarget,
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
            if(!ftRef->frame()->isVisible(ftRef->position(),std::max(_patchSize,2U),level))
            {
                throw pd::Exception("Feature [" + std::to_string(ftRef->id()) + "] is not fully visible in reference frame.");

            }

            const auto frameRef = ftRef->frame();
            const auto mat = frameRef->grayImage(level);
            const auto pCamera = frameRef->world2frame(_p3d);
            const auto J_xyz2uv = frameRef->camera()->J_xyz2uv(pCamera);

            VLOG(5) << "J_xyz2uv =\n " << J_xyz2uv;

            utils::throw_if_nan(J_xyz2uv,"Point Jac.",ftRef);

            const auto pImg = _ftRef->position();
            _jacobian.resize(6,patchSize*patchSize);
            _patchRef.resize (_patchSize,_patchSize);
            int idxPixel = 0;
            for (int i = 0; i < _patchSize; i++)
            {
                for (int j = 0; j < _patchSize; j++)
                {
                    _patchRef(i,j) = algorithm::bilinearInterpolation(mat,
                                                                      (pImg.x() - (double)_patchSizeHalf + (double)j ) *_scale,
                                                                      (pImg.y() - (double)_patchSizeHalf + (double)i ) *_scale);

                    const double fu1 = algorithm::bilinearInterpolation(mat,
                                                                        (pImg.x() - (double)_patchSizeHalf + (double)j + 1.0) *_scale,
                                                                        (pImg.y() - (double)_patchSizeHalf + (double)i  ) *_scale);
                    const double fu0 = algorithm::bilinearInterpolation(mat,
                                                                        (pImg.x() - (double)_patchSizeHalf + (double)j - 1.0  ) *_scale,
                                                                        (pImg.y() - (double)_patchSizeHalf + (double)i ) *_scale);
                    const double fv1 = algorithm::bilinearInterpolation(mat,
                                                                        (pImg.x() - (double)_patchSizeHalf + (double)j) *_scale,
                                                                        (pImg.y() - (double)_patchSizeHalf + (double)i + 1.0) *_scale);
                    const double fv0 = algorithm::bilinearInterpolation(mat,
                                                                        (pImg.x() - (double)_patchSizeHalf + (double)j) *_scale,
                                                                        (pImg.y() - (double)_patchSizeHalf + (double)i - 1.0) *_scale);

                    const double du = 0.5 * (fu1 - fu0);
                    const double dv = 0.5 * (fv1 - fv0);

                    _jacobian.col(idxPixel++) = (du*J_xyz2uv.row(0) + dv*J_xyz2uv.row(1))*(frameRef->camera()->focalLength() / _scale);

                }
            }


            utils::throw_if_nan(_jacobian,"Jac.",ftRef);

            VLOG(5) << "Photometric Loss for feature [ " << ftRef->id() << " ]:"
                    << " \n W = " << _patchRef
                    << " \n J = " << _jacobian;
            //this->set_num_residuals(_patchArea);

        }

        template <int patchSize>
        bool ImageAlignment<patchSize>::Cost::Evaluate(double const* const* parameters,double* residuals,double** jacobians) const {

            const Eigen::Map<const Eigen::Matrix<double, Sophus::SE3d::DoF, 1>> pose(*parameters);
            const auto T = Sophus::SE3d::exp(pose);
            const auto pCamera = T * _p3d;
            const auto pImg = _frameTarget->camera2image(pCamera);

            VLOG(5) << "Reprojection: " << pImg.transpose();

            if (jacobians != nullptr)
            {
                Eigen::Map<Eigen::Matrix<double, Sophus::SE3d::DoF,patchSize*patchSize,Eigen::ColMajor>> j(jacobians[0]);

                j = _jacobian;
                VLOG(5) << "J:\n" << _jacobian;

            }

            if ( _frameTarget->isVisible(pImg,_patchSizeHalf,_level))
            {
                int idxPixel = 0;
                for (int i = 0; i < _patchSize; i++)
                {
                    for (int j = 0; j < _patchSize; j++)
                    {
                        const double x = (pImg.x() - (double)_patchSizeHalf + (double) j ) *_scale;
                        const double y = (pImg.y() - (double)_patchSizeHalf + (double) i ) *_scale;
                        const double grayTarget =     algorithm::bilinearInterpolation(_frameTarget->grayImage(_level),
                                                                                 x,
                                                                                 y);

                        const double res = _patchRef(i,j) - grayTarget;
                        VLOG(5) << "Residual at (" << x << "," << y << "): " << grayTarget << "-" << _patchRef(i,j) << "=" << res;

                        residuals[idxPixel++] = res;
                    }

                }

            }else{
                int idxPixel = 0;
                for (int i = 0; i < _patchSize; i++)
                {
                    for (int j = 0; j < _patchSize; j++)
                    {
                         VLOG(5) << _ftRef->id() << " not visible.";

                        residuals[idxPixel++] = 0;
                    }

                }

            }
            return true;

        }


    }}