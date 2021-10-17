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
#include "utils/visuals.h"

#include "solver/LeastSquaresSolver.h"
#include "ImageAlignmentSparse.h"

namespace pd{ namespace vision{



class CostFeature
{
private:
    Eigen::MatrixXd _jacobian;
    Eigen::MatrixXd _patchRef;
    const Eigen::Vector3d _p3d;
    const Feature2D::ShConstPtr _ftRef;
    const Frame::ShConstPtr _frameTarget;
    const std::uint32_t _patchSize;
    const std::uint32_t _patchArea;
    const float _patchSizeHalf;
    const std::uint32_t _level;
    const double _scale;
    const int _idx;
public:
    CostFeature (int idx,
                 int patchSize,
                 Feature2D::ShConstPtr ftRef,
                 Frame::ShConstPtr frameTarget,
                 uint32_t level)
    : _idx(idx)
    , _ftRef(ftRef)
    , _frameTarget(frameTarget)
    , _p3d(ftRef->point()->position())
    , _patchSize(patchSize)
    , _patchArea(patchSize *patchSize)
    , _level(level)
    , _scale( 1.0/(1U<<level))
    , _patchSizeHalf(patchSize/2.0){

        if (!ftRef->point()) {
            throw pd::Exception("Feature [" + std::to_string(ftRef->id()) + "] does not have corresponding 3D point.");
        }
        if (!ftRef->frame()->isVisible(ftRef->position(), std::max(_patchSize, 2U), level)) {
            throw pd::Exception(
                    "Feature [" + std::to_string(ftRef->id()) + "] is not fully visible in reference frame.");

        }

        const auto frameRef = ftRef->frame();
        const auto mat = frameRef->grayImage(level);
        const auto pCamera = frameRef->world2frame(_p3d);
        const auto J_xyz2uv = frameRef->camera()->J_xyz2uv(pCamera, _scale);

        VLOG(5) << "J_xyz2uv =\n " << J_xyz2uv;

        utils::throw_if_nan(J_xyz2uv, "Point Jac.", ftRef);

        const auto pImg = _ftRef->position();
        _jacobian.resize(patchSize * patchSize, Sophus::SE3d::DoF);
        _patchRef.resize(_patchSize, _patchSize);
        int idxPixel = 0;
        for (int i = 0; i < _patchSize; i++) {
            for (int j = 0; j < _patchSize; j++) {
                _patchRef(i, j) = algorithm::bilinearInterpolation(mat,
                                                                   (pImg.x() - (double) _patchSizeHalf + (double) j) *
                                                                   _scale,
                                                                   (pImg.y() - (double) _patchSizeHalf + (double) i) *
                                                                   _scale);

                const double fu1 = algorithm::bilinearInterpolation(mat,
                                                                    (pImg.x() - (double) _patchSizeHalf + (double) j +
                                                                     1.0) * _scale,
                                                                    (pImg.y() - (double) _patchSizeHalf + (double) i) *
                                                                    _scale);
                const double fu0 = algorithm::bilinearInterpolation(mat,
                                                                    (pImg.x() - (double) _patchSizeHalf + (double) j -
                                                                     1.0) * _scale,
                                                                    (pImg.y() - (double) _patchSizeHalf + (double) i) *
                                                                    _scale);
                const double fv1 = algorithm::bilinearInterpolation(mat,
                                                                    (pImg.x() - (double) _patchSizeHalf + (double) j) *
                                                                    _scale,
                                                                    (pImg.y() - (double) _patchSizeHalf + (double) i +
                                                                     1.0) * _scale);
                const double fv0 = algorithm::bilinearInterpolation(mat,
                                                                    (pImg.x() - (double) _patchSizeHalf + (double) j) *
                                                                    _scale,
                                                                    (pImg.y() - (double) _patchSizeHalf + (double) i -
                                                                     1.0) * _scale);

                const double du = 0.5 * (fu1 - fu0);
                const double dv = 0.5 * (fv1 - fv0);

                _jacobian.row(idxPixel++) = (du * J_xyz2uv.row(0) + dv * J_xyz2uv.row(1));

            }
        }


        utils::throw_if_nan(_jacobian, "Jac.", ftRef);

        VLOG(5) << "Photometric Loss for feature [ " << ftRef->id() << " ]:"
                << " \n W = " << _patchRef
                << " \n J = " << _jacobian;
    }

    bool computeJacobian(Eigen::MatrixXd &jacobian)
    {
        for (int i = 0; i < (_patchArea); i++)
        {
            jacobian.row(_idx + i) = _jacobian.row(i);

        }
        return true;
    }

    bool computeResidual(const Sophus::SE3d& pose, Eigen::MatrixXd& residual, Eigen::MatrixXd& weights)
    {
        const Eigen::MatrixXd pCamera = pose * _p3d;

        const Eigen::Vector2d pImg = _frameTarget->camera2image(pCamera);
        VLOG(5) << "Reprojection: " << pImg.transpose();

        if ( _frameTarget->isVisible(pImg,_patchSizeHalf,_level))
        {
            int idxPixel = 0;
            for (int i = 0; i < _patchSize; i++)
            {
                for (int j = 0; j < _patchSize; j++)
                {
                    const double x = (pImg.x() - _patchSizeHalf + (double) j ) *_scale;
                    const double y = (pImg.y() - _patchSizeHalf + (double) i ) *_scale;
                    const double patchTargetij = algorithm::bilinearInterpolation(_frameTarget->grayImage(_level),
                                                                                  x,
                                                                                  y);

                    const double res = _patchRef(i,j) - patchTargetij;
                    VLOG(5) << "Residual at (" << x << "," << y << "): " << patchTargetij << "-" << _patchRef(i,j) << "=" << res;

                    if (std::isnan(res))
                    {
                        throw pd::Exception("NaN for: " + std::to_string(_ftRef->id()));
                    }
                    residual(_idx + idxPixel) = res;
                    weights(_idx + idxPixel) = 1.0;
                    idxPixel++;
                }
            }
        }else{
            int idxPixel = 0;
            for (int i = 0; i < _patchSize; i++)
            {
                for (int j = 0; j < _patchSize; j++)
                {
                    weights(_idx + idxPixel++) = 0.0;
                }
            }
        }
        utils::throw_if_nan(residual.row(_idx),"Residuals",_ftRef);

        return true;


    }
};
class CostTotal
{
    std::vector<std::shared_ptr<CostFeature>> _costFeatures;
    Eigen::MatrixXd _jacobian;
public:
    CostTotal(int patchSize, Frame::ShConstPtr referenceFrame, Frame::ShConstPtr targetFrame, int level, int nVisiblePoints)
    {

        if (nVisiblePoints <= Sophus::SE3d::DoF)
        {
            throw pd::Exception(std::to_string(nVisiblePoints) + " points are not enough. Need at least: " + std::to_string(Sophus::SE3d::DoF));
        }
        Sophus::SE3d T = algorithm::computeRelativeTransform(referenceFrame->pose(), targetFrame->pose());

        _costFeatures.resize(nVisiblePoints);
        _jacobian.conservativeResize(nVisiblePoints*patchSize*patchSize,Sophus::SE3d::DoF);

        int idxF = 0;
        for ( const auto& f : referenceFrame->features() )
        {
            if ( f->point() &&  f->frame()->isVisible(f->position(), std::max(patchSize, 2), level))
            {
                auto costFeature = std::make_shared<CostFeature> (idxF * (patchSize*patchSize), patchSize, f, targetFrame, level);
                costFeature->computeJacobian(_jacobian);
                _costFeatures[idxF++] = costFeature;
            }
        }

    }

    bool computeResidual(const Eigen::MatrixXd& x, Eigen::MatrixXd& residual, Eigen::MatrixXd& weights) const {

    const Sophus::SE3d pose = Sophus::SE3d::exp(x);

    for (const auto& cf : _costFeatures)
    {
        cf->computeResidual(pose,residual,weights);
    }

    return true;

}
    bool computeJacobian(const Eigen::MatrixXd& x, Eigen::MatrixXd& jacobian)
{
    jacobian = _jacobian;
    return true;
}
};

    void ImageAlignmentSparse::align(Frame::ShConstPtr referenceFrame, Frame::ShPtr targetFrame) const
    {
        for (int level = _levelMax; level >= _levelMin; --level)
        {
            Sophus::SE3d T = algorithm::computeRelativeTransform(referenceFrame->pose(), targetFrame->pose());
            Eigen::MatrixXd posev6d = T.log();
            VLOG(4) << "IA init: " << " Level: " << level  << " #Features: " << referenceFrame->features().size();

            int nVisiblePoints = 0;
            for ( const auto& f : referenceFrame->features() )
            {
                if ( f->point() && f->frame()->isVisible(f->position(), std::max(_patchSize, 2), level))
                {
                    nVisiblePoints++;
                }
            }

            auto cost = std::make_shared<CostTotal>(_patchSize,referenceFrame,targetFrame,level,nVisiblePoints);
            auto lls = std::make_shared<LeastSquaresSolver>(
                    [&](const Eigen::MatrixXd& x, Eigen::MatrixXd& residual, Eigen::MatrixXd& weights) { return cost->computeResidual(x,residual,weights);},
                    [&](const Eigen::MatrixXd& x, Eigen::MatrixXd& jacobian) { return cost->computeJacobian(x,jacobian);},
                    [&](const Eigen::MatrixXd& dx, Eigen::MatrixXd& x) { x = (Sophus::SE3d::exp(x) * Sophus::SE3d::exp(-dx)).log(); return true;},
                    nVisiblePoints * _patchSize * _patchSize,
                    Sophus::SE3d::DoF,
                    0.0001,
                    0.0001,
                    100
            );

            lls->solve(posev6d);

            targetFrame->setPose(Sophus::SE3d::exp(posev6d)*referenceFrame->pose());

            Log::getImageLog("image_alignment")->append(&vis::drawReprojection,std::shared_ptr<const Frame>(referenceFrame),std::shared_ptr<const Frame>(targetFrame),(int)(_patchSize/2));

            if (VLOG_IS_ON(4))
            {
            //    VLOG(4) << summary.FullReport();

            }else{
            //    VLOG(3) << summary.BriefReport();

            }
        }

    }


    ImageAlignmentSparse::ImageAlignmentSparse(uint32_t patchSize, uint32_t levelMax, uint32_t levelMin)
    : _levelMax(levelMax)
    , _levelMin(levelMin)
    , _patchSize(patchSize)
    {}

    }
}
