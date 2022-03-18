
#include "MotionPrediction.h"
#include "utils/utils.h"
#define LOG_MOTION_PREDICTION(level) CLOG(level,"motion_prediction")
namespace pd::vision{

        MotionPrediction::ShPtr MotionPrediction::make(const std::string& model)
        {
                if ( model == "NoMotion")
                {
                        return std::make_shared<MotionPredictionNoMotion>();
                }else if (model == "ConstantMotion")
                {
                        return std::make_shared<MotionPredictionConstant>();

                }else{
                        LOG_MOTION_PREDICTION( WARNING ) << "Unknown motion model! Falling back to constant motion model.";
                        return std::make_shared<MotionPredictionConstant>();
                }
        }


        void MotionPredictionConstant::update(PoseWithCovariance::ConstShPtr pose, Timestamp timestamp)
        {
                const double dT = ((double)timestamp - (double) _lastT)/1e9;
                _speed = algorithm::computeRelativeTransform(_lastPose->pose(),pose->pose()).log()/dT;
                _lastPose = pose;
                _lastT = timestamp;
        }
        PoseWithCovariance::UnPtr MotionPredictionConstant::predict(Timestamp timestamp) const
        {
                const double dT = ((double)timestamp - (double) _lastT)/1e9;
                const SE3d predictedRelativePose = SE3d::exp(_speed * dT);
                return std::make_unique<PoseWithCovariance>(predictedRelativePose * _lastPose->pose(),MatXd::Identity(6,6));
        }
       
}