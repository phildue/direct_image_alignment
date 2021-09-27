#include <utility>
#include <iostream>

#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <ceres/rotation.h>

#include "core/Camera.h"
#include "core/algorithm.h"
#include "core/Point3D.h"
#include "core/Feature2D.h"

#include "utils/utils.h"

#include "SE3Parameterization.h"
#include "ImageAlignment.h"

namespace pd{ namespace vision{

        template<int patchSize>
        class ImageAlignmentCeres : public ImageAlignment<patchSize>
        {
        public:
            ImageAlignmentCeres(uint32_t levelMax, uint32_t levelMin):ImageAlignment<patchSize>(levelMax,levelMin){}
            void align(Frame::ShConstPtr referenceFrame, Frame::ShConstPtr targetFrame) const override;

            struct Cost : ceres::SizedCostFunction<patchSize*patchSize,Sophus::SE3d::DoF>{
                Eigen::Matrix<double, Sophus::SE3d::DoF,Eigen::Dynamic,Eigen::ColMajor> _jacobian;
                Eigen::MatrixXd _patchRef;
                const Eigen::Vector3d  _p3d;
                const Feature2D::ShConstPtr _ftRef;
                const Frame::ShConstPtr  _frameTarget;
                const std::uint32_t _patchSize;
                const std::uint32_t  _patchArea;
                const float  _patchSizeHalf;
                const std::uint32_t  _level;
                const double _scale;
                Cost (Feature2D::ShConstPtr ftRef,
                      Frame::ShConstPtr frameTarget,
                      std::uint32_t level
                );
                bool Evaluate(double const* const* pose_raw,double* residuals,double** jacobians_raw) const ;
            };
        };

        template <int patchSize>
        ImageAlignmentCeres<patchSize>::Cost::Cost (Feature2D::ShConstPtr ftRef,
                                                    Frame::ShConstPtr frameTarget,
                                                    uint32_t level)
                : _ftRef(ftRef)
                , _frameTarget(frameTarget)
                , _p3d(ftRef->point()->position())
                , _patchSize(patchSize)
                , _patchArea(patchSize*patchSize)
                , _level(level)
                , _scale(1.0/(1U<<level))
                , _patchSizeHalf(patchSize/2.0){

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
            const auto J_xyz2uv = frameRef->camera()->J_xyz2uv(pCamera,_scale);

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

                    _jacobian.col(idxPixel++) = (du*J_xyz2uv.row(0) + dv*J_xyz2uv.row(1));

                }
            }


            utils::throw_if_nan(_jacobian,"Jac.",ftRef);

            VLOG(5) << "Photometric Loss for feature [ " << ftRef->id() << " ]:"
                    << " \n W = " << _patchRef
                    << " \n J = " << _jacobian;
            //this->set_num_residuals(_patchArea);

        }

        template <int patchSize>
        bool ImageAlignmentCeres<patchSize>::Cost::Evaluate(double const* const* pose_raw,double* residuals,double** jacobians_raw) const {

            const Eigen::Map<const Sophus::SE3d> pose(*pose_raw);
            const Eigen::Vector3d pCamera = pose * _p3d;
            const Eigen::Vector2d pImg = _frameTarget->camera2image(pCamera);

            VLOG(5) << "Reprojection: " << pImg.transpose();

            if (jacobians_raw != nullptr && jacobians_raw[0] != nullptr)
            {
                Eigen::Map<Eigen::Matrix<double, Sophus::SE3d::DoF,patchSize*patchSize,Eigen::ColMajor>> j(jacobians_raw[0]);

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
                        const double x = (pImg.x() - _patchSizeHalf + (double) j ) *_scale;
                        const double y = (pImg.y() - _patchSizeHalf + (double) i ) *_scale;
                        const double patchTargetij =     algorithm::bilinearInterpolation(_frameTarget->grayImage(_level),
                                                                                          x,
                                                                                          y);

                        const double res = _patchRef(i,j) - patchTargetij;
                        VLOG(5) << "Residual at (" << x << "," << y << "): " << patchTargetij << "-" << _patchRef(i,j) << "=" << res;

                        if (std::isnan(res))
                        {
                            throw pd::Exception("NaN for: " + std::to_string(_ftRef->id()));
                        }
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
            const Eigen::Map<const Eigen::Matrix<double,patchSize*patchSize, 1>> residualsMat(residuals);
            utils::throw_if_nan(residualsMat,"Residuals",_ftRef);

            return true;

        }

    template<int patchSize>
    void ImageAlignmentCeres<patchSize>::align(Frame::ShConstPtr referenceFrame, Frame::ShConstPtr targetFrame) const
    {
        for (int level = this->_levelMax; level >= this->_levelMin; --level)
        {
            Sophus::SE3d T = algorithm::computeRelativeTransform(referenceFrame->pose(), targetFrame->pose());
            Eigen::Vector6d posev6d = T.log();
            VLOG(4) << "IA init: " << " Level: " << level  << " #Features: " << referenceFrame->features().size();

            ceres::Problem problem;
            problem.AddParameterBlock(T.data(),Sophus::SE3d::num_parameters,new SE3qtParam());

            for ( int idxF = 0; idxF < referenceFrame->features().size(); idxF++)
            {
                const Feature2D::ShConstPtr f = referenceFrame->features()[idxF];
                if ( f->point() )
                {
                    if ( referenceFrame->isVisible(f->position(),patchSize, level))
                    {
                        auto cost = new ImageAlignmentCeres<patchSize>::Cost(f,targetFrame,level);
                        problem.AddResidualBlock(cost, nullptr,T.data());

                    }
                }
            }
            VLOG(4) << "Setup IA with #Parameters: " << problem.NumParameters() << ", #Residuals: " << problem.NumResiduals();
            ceres::Solver::Options options{};
            options.linear_solver_type = ceres::DENSE_SCHUR;

            options.minimizer_progress_to_stdout = VLOG_IS_ON(3);

            ceres::Solver::Summary summary{};
            ceres::Solve(options, &problem, &summary);
            targetFrame->setPose(Sophus::SE3d::exp(posev6d)*referenceFrame->pose());

            Log::logReprojection(referenceFrame,targetFrame,patchSize/2,4);

            if (VLOG_IS_ON(4))
            {
            //    VLOG(4) << summary.FullReport();

            }else{
            //    VLOG(3) << summary.BriefReport();

            }
        }

    }



}}
