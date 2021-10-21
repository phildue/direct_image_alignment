#include <utility>
#include <iostream>

#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <ceres/rotation.h>

#include "core/Camera.h"
#include "core/algorithm.h"
#include "core/Point3D.h"
#include "core/Feature2D.h"
#include "core/Frame.h"

#include "utils/utils.h"

#include "solver/LeastSquaresSolver.h"
#include "ImageAlignmentDense.h"
#include "utils/visuals.h"

namespace pd{ namespace vision{



class OptimizerHandle
{
    const int _stepSizeX, _stepSizeY;
    const std::uint32_t _level;
    const double _scale;
    Eigen::MatrixXd _jacobian;
    Eigen::Matrix<double, 3, Eigen::Dynamic> _pointsCamRef;
    const Frame::ShConstPtr _frameTarget;
    const FrameRGBD::ShConstPtr _frameRef;
    const int _nPixels;



public:
    OptimizerHandle(FrameRGBD::ShConstPtr referenceFrame, Frame::ShConstPtr targetFrame, int level, int stepSizeX, int stepSizeY, int nPixels)
    :_stepSizeX(stepSizeX)
    ,_stepSizeY(stepSizeY)
    , _level(level)
    , _scale( 1.0/(1U<<level))
    , _frameRef(referenceFrame)
    , _frameTarget(targetFrame)
    , _nPixels(nPixels)
    {
        IMAGE_ALIGNMENT( INFO ) << "Size: " << referenceFrame->width(_level) << " x " << referenceFrame->height(_level)
        << " Steps: " << _stepSizeX << " x " << _stepSizeY
        << " #Pixels: " << nPixels; 

        _jacobian.conservativeResize(nPixels,Sophus::SE3d::DoF);
        _jacobian.setZero();
        _pointsCamRef.conservativeResize(Eigen::NoChange,nPixels);
        _pointsCamRef.setZero();
        Eigen::MatrixXd jacobianImageX(referenceFrame->height(_level),referenceFrame->width(_level));
        Eigen::MatrixXd jacobianImageY(referenceFrame->height(_level),referenceFrame->width(_level));
        Eigen::MatrixXd jacobianImageZ(referenceFrame->height(_level),referenceFrame->width(_level));
        Eigen::MatrixXd jacobianImageRx(referenceFrame->height(_level),referenceFrame->width(_level));
        Eigen::MatrixXd jacobianImageRy(referenceFrame->height(_level),referenceFrame->width(_level));
        Eigen::MatrixXd jacobianImageRz(referenceFrame->height(_level),referenceFrame->width(_level));
        
        const auto& mat = referenceFrame->grayImage(_level);
        int idxPixel = 0;
        for (int v = 1; v < referenceFrame->height(_level)-1; v += _stepSizeY)
        {
            for (int u = 1; u < referenceFrame->width(_level)-1; u += _stepSizeX)
            {
                const double depth = referenceFrame->depthMap(_level)(v,u);

                if ( !std::isnan(depth) && depth >= 0.1 && !std::isinf(depth))
                {
                    _pointsCamRef.col(idxPixel) = referenceFrame->image2camera({u,v},depth);

                    utils::throw_if_nan(_pointsCamRef.col(idxPixel), "3D Point");

                    const auto J_xyz2uv = referenceFrame->camera()->J_xyz2uv(_pointsCamRef.col(idxPixel), _scale);

                    IMAGE_ALIGNMENT( DEBUG ) << "scale: " << _scale << "p3d:" << _pointsCamRef.col(idxPixel).transpose() 
                    << " \nJ_xyz2uv =\n " << J_xyz2uv;

                    utils::throw_if_nan(J_xyz2uv, "J_xyz2uv");

                    const double fu0 = mat(v, (u - 1));
                    const double fu1 = mat(v, (u + 1));
                    const double fv1 = mat((v + 1), u);
                    const double fv0 = mat((v - 1), u);

                    const double du = 0.5 * (fu1 - fu0);
                    const double dv = 0.5 * (fv1 - fv0);

                    _jacobian.row(idxPixel) = (du * J_xyz2uv.row(0) + dv * J_xyz2uv.row(1));
                    jacobianImageX(v,u) = _jacobian.row(idxPixel)(0);
                    jacobianImageY(v,u) = _jacobian.row(idxPixel)(1);
                    jacobianImageZ(v,u) = _jacobian.row(idxPixel)(2);
                    jacobianImageRx(v,u) = _jacobian.row(idxPixel)(3);
                    jacobianImageRy(v,u) = _jacobian.row(idxPixel)(4);
                    jacobianImageRz(v,u) = _jacobian.row(idxPixel)(5);
                    
                    utils::throw_if_nan(_jacobian.row(idxPixel), "J_Feature");
                    idxPixel++;
                }
            }
        }
        cv::Mat m = vis::drawAsImage(jacobianImageX);
         Log::getImageLog("Jx")->append(m);
         Log::getImageLog("Jy")->append(&vis::drawAsImage,jacobianImageY);
         Log::getImageLog("Jz")->append(&vis::drawAsImage,jacobianImageZ);
         Log::getImageLog("Jrx")->append(&vis::drawAsImage,jacobianImageRx);
         Log::getImageLog("Jry")->append(&vis::drawAsImage,jacobianImageRy);
         Log::getImageLog("Jrz")->append(&vis::drawAsImage,jacobianImageRz);

    }

    bool computeResidual(const Eigen::VectorXd& x, Eigen::VectorXd& residual, Eigen::VectorXd& weights) const
    {

        const Sophus::SE3d pose = Sophus::SE3d::exp(x);

       //TODO const Eigen::Matrix<double,3,Eigen::Dynamic> pointsCamTarget = pose * _pointsCamRef;
       Eigen::MatrixXd residualImage(_frameRef->height(_level),_frameRef->width(_level));
        int idxPoint = 0;
        for (int v = 1; v < _frameRef->height(_level)-1; v += _stepSizeY)
        {
            for (int u = 1; u < _frameRef->width(_level)-1; u += _stepSizeX)
            {
                const double depth = _frameRef->depthMap(_level)(v,u);

                if ( !std::isnan(depth) && depth >= 0.1 && !std::isinf(depth))
                {
                    const Eigen::Vector3d pointsCamTarget = pose * _pointsCamRef.col(idxPoint);
                    const Eigen::Vector2d pImgTarget = _frameTarget->camera2image(pointsCamTarget);

                    if ( _frameTarget->isVisible(pImgTarget,1,_level))
                    {
                        const double intensityRef =_frameRef->grayImage(_level)(v,u);
                        const double intensityTarget = algorithm::bilinearInterpolation(_frameTarget->grayImage(_level),pImgTarget.x()*_scale,pImgTarget.y()*_scale);
                        const double res = intensityTarget - intensityRef;

                        if (std::isnan(res))
                        {
                            throw pd::Exception("Residual is NaN for: (" + std::to_string( u ) + "," + std::to_string(v) + ")");
                        }

                        residual(idxPoint) = res;
                        residualImage(v,u) = res;
                        weights(idxPoint) = 1.0;
                    }else{
                        residual(idxPoint) = 0.0;
                        weights(idxPoint) = 0.0;
                    }
                    idxPoint++;
                }
            }   
        }
         Log::getImageLog("Residual")->append(vis::drawAsImage,residualImage);

        return true;
    }

    bool computeJacobian(const Eigen::VectorXd& x, Eigen::MatrixXd& jacobian)
    {
        jacobian = _jacobian;
        return true;
    }
};

    void ImageAlignmentDense::align(FrameRGBD::ShConstPtr referenceFrame, Frame::ShPtr targetFrame) const
    {
         Log::getImageLog("RefFrame")->append(&vis::drawFrame,std::shared_ptr<const Frame>(referenceFrame));
         Log::getImageLog("TargetFrame")->append(&vis::drawFrame,std::shared_ptr<const Frame>(targetFrame));

        for (int level = _levelMax; level >= _levelMin; --level)
        {
            Sophus::SE3d T = algorithm::computeRelativeTransform(referenceFrame->pose(), targetFrame->pose());
            Eigen::VectorXd posev6d = T.log();
            IMAGE_ALIGNMENT( INFO )<< "Starting Level: " << level << " Init: " << posev6d.transpose();

            const int nPixels = countValidPoints(referenceFrame,level);

            auto cost = std::make_shared<OptimizerHandle>(referenceFrame,targetFrame,level,_stepSizeX,_stepSizeY,nPixels);
            auto lls = std::make_shared<LeastSquaresSolver>(
                    [&](const Eigen::VectorXd& x, Eigen::VectorXd& residual, Eigen::VectorXd& weights) { return cost->computeResidual(x,residual,weights);},
                    [&](const Eigen::VectorXd& x, Eigen::MatrixXd& jacobian) { return cost->computeJacobian(x,jacobian);},
                    [&](const Eigen::VectorXd& dx, Eigen::VectorXd& x) { x = (Sophus::SE3d::exp(x) * Sophus::SE3d::exp(-dx)).log(); return true;},
                    nPixels,
                    Sophus::SE3d::DoF,
                    1e-2,
                    1e-16,
                    30
            );

            lls->solve(posev6d);
            IMAGE_ALIGNMENT( INFO )<< "Result: " << posev6d.transpose();

            targetFrame->setPose(Sophus::SE3d::exp(posev6d)*referenceFrame->pose());

            }

    }


        ImageAlignmentDense::ImageAlignmentDense(uint32_t levelMax, uint32_t levelMin, uint32_t stepSizeX, uint32_t stepSizeY)
    : _levelMax(levelMax)
    , _levelMin(levelMin)
    , _stepSizeX(stepSizeX)
    , _stepSizeY(stepSizeY)
    {
        Log::get("image_alignment");
    }

    int ImageAlignmentDense::countValidPoints(FrameRGBD::ShConstPtr referenceFrame, int level) const
    {
        int nPixel = 0;
        for (int v = 1; v < referenceFrame->height(level)-1; v += _stepSizeY)
        {
            for (int u = 1; u < referenceFrame->width(level)-1; u += _stepSizeX)
            {
                const double depth = referenceFrame->depthMap(level)(v,u);

                if ( !std::isnan(depth) && depth >= 0.1 && !std::isinf(depth))
                {   
                    nPixel++;
                }
            }
        }
        return nPixel;
    }

    }
}
