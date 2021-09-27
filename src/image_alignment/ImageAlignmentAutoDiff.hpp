#include <utility>
#include <iostream>

#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <ceres/rotation.h>

#include "utils/Log.h"
#include "ImageAlignment.h"
#include "core/Camera.h"
#include "core/algorithm.h"
#include "core/Point3D.h"
#include "core/Feature2D.h"
#include "utils/Exceptions.h"
#include "utils/utils.h"
#include "SE3Parameterization.h"

namespace pd{ namespace vision{

        template<int patchSize>
        class ImageAlignmentAutoDiff : public ImageAlignment<patchSize>
        {
        public:
            ImageAlignmentAutoDiff(uint32_t levelMax, uint32_t levelMin):ImageAlignment<patchSize>(levelMax,levelMin){}
            void align(Frame::ShConstPtr referenceFrame, Frame::ShConstPtr targetFrame) const override;
        };


        template<int patchSize>
    class PhotometricError{
        const Feature2D::ShConstPtr _ftRef;
        const Frame::ShConstPtr _frameTarget;
        const int _patchSize;
        const int _patchSizeHalf;
        const int _level;
        const double _scale;
        const double _cx,_cy,_f;
        Eigen::MatrixXd _patchRef;
        Eigen::Vector3d  _p3d;
        ceres::Grid2D<std::uint8_t> _img;
        ceres::BiCubicInterpolator<ceres::Grid2D<std::uint8_t >> _interpolator;

    public:
        PhotometricError(Feature2D::ShConstPtr ftRef,
                         Frame::ShConstPtr frameTarget,
                         uint32_t level)
                         : _ftRef(ftRef)
                         , _frameTarget(std::move(frameTarget))
                         , _patchSize(patchSize)
                         , _patchSizeHalf(patchSize/2)
                         , _level(level)
                         , _scale(1.0/(1U<<level))
                         , _f(ftRef->frame()->camera()->focalLength())
                , _cx(ftRef->frame()->camera()->principalPoint().x())
                , _cy(ftRef->frame()->camera()->principalPoint().y())
                , _img((std::uint8_t*)_frameTarget->grayImage(level).data(),0,_frameTarget->height(level),0,_frameTarget->width(level))
                , _interpolator(_img)
        {
            if(!ftRef->point())
            {
                throw pd::Exception("Feature [" + std::to_string(ftRef->id()) + "] does not have corresponding 3D point.");
            }
            if(!ftRef->frame()->isVisible(ftRef->position(),_patchSize,level))
            {
                throw pd::Exception("Feature [" + std::to_string(ftRef->id()) + "] is not fully visible in reference frame.");

            }
            _p3d = ftRef->point()->position();

             computeReferencePatch();

            VLOG(4) << " W = \n" << _patchRef;
        }
        void computeReferencePatch()
        {
            _patchRef.resize(_patchSize,_patchSize);
            const auto& mat = _ftRef->frame()->grayImage(_level);
            for (int r = 0; r < _patchSize; r++)
            {
                for (int c = 0; c < _patchSize; c++) {
                    _patchRef(r, c) = algorithm::bilinearInterpolation(mat,
                                                                   (_ftRef->position().x() - _patchSizeHalf + c) * _scale,
                                                                   (_ftRef->position().y() - _patchSizeHalf + r) * _scale);
                }
            }
        }
        template <typename T>
        bool operator()(const T* const pose, T* sResiduals) const {

            const Eigen::Map<const Sophus::SE3<T> > T_wa(pose);
            Eigen::Map<Eigen::Matrix<T,patchSize*patchSize,1> > residuals(sResiduals);

            const Eigen::Matrix<T,3,1> p3dCcs = T_wa * _p3d.cast<T>();
            //TODO check z > 0?

            Eigen::Matrix<T,3,1> pProj = _frameTarget->camera()->K() * p3dCcs;
            pProj /= pProj.z();

            //TODO check image boundaries?

            int idxPixel(0);

            for (int r = 0; r < _patchSize; r ++)
            {
                for (int c = 0; c < _patchSize; c ++)
                {
                    T imgX = (pProj.x() + ((double)r -(double)_patchSizeHalf))*_scale;
                    T imgY = (pProj.y() + ((double)c -(double)_patchSizeHalf))*_scale;
                    T f;
                    _interpolator.Evaluate(imgX, imgY, &f);
                    residuals(idxPixel++,0) = f - (T)_patchRef(r,c);
                }
            }

            return true;


        }



        };


    template<int patchSize>
    void ImageAlignmentAutoDiff<patchSize>::align(Frame::ShConstPtr referenceFrame, Frame::ShConstPtr targetFrame) const{

        for (int level = this->_levelMax; level >= this->_levelMin; --level)
        {
            Sophus::SE3d pose = targetFrame->pose();
            VLOG(4) << "IA init: " << " Level: " << level  << " #Features: " << referenceFrame->features().size();

            ceres::Problem problem;
            for ( int idxF = 0; idxF < referenceFrame->features().size(); idxF++)
            {
                const auto& f = referenceFrame->features()[idxF];
                if ( f->point() )
                {
                    if ( referenceFrame->isVisible(f->position(),patchSize, level))
                    {
                        auto cost = new ceres::AutoDiffCostFunction<PhotometricError<patchSize>,patchSize*patchSize,
                                Sophus::SE3d::num_parameters>(new PhotometricError<patchSize>(f,targetFrame,level));
                        problem.AddParameterBlock(pose.data(),Sophus::SE3d::num_parameters,new SE3atParam());
                        problem.AddResidualBlock(cost, nullptr,pose.data());

                    }
                }
            }
            ceres::Solver::Options options{};
            options.linear_solver_type = ceres::DENSE_SCHUR;
            options.minimizer_progress_to_stdout = true;

            ceres::Solver::Summary summary{};
            VLOG(4) << "Setup IA with #Parameters: " << problem.NumParameters() << ", #Residuals: " << problem.NumResiduals();
            ceres::Solve(options, &problem, &summary);
            targetFrame->setPose(pose);
            Log::logReprojection(referenceFrame,targetFrame,patchSize/2,4);
            if (VLOG_IS_ON(4))
            {
//                VLOG(4) << summary.FullReport();

            }else{
  //              VLOG(3) << summary.BriefReport();

            }
        }
    }



    }
}
