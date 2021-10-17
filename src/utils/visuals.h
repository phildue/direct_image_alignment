//
// Created by phil on 07.08.21.
//

#ifndef VSLAM_VISUALS_H
#define VSLAM_VISUALS_H

#include <memory>

#include <opencv2/core/mat.hpp>
#include "core/types.h"


namespace pd{ namespace vision{
    class Frame;
    class FrameRGBD;
    
    namespace vis{
        
        cv::Mat drawFrame(std::shared_ptr<const Frame> frame);
        cv::Mat drawFeatures(std::shared_ptr<const Frame> frame, int radius = 3, bool gradient = true);

        cv::Mat drawAsImage(Eigen::MatrixXd residual);
        //static void logJacobianImage(int iteration, const Eigen::MatrixXd& jacobian);

        cv::Mat drawFeaturesWithPoints(std::shared_ptr<const Frame> frame, int radius = 3);
        cv::Mat drawReprojection(std::shared_ptr<const Frame> frame0,std::shared_ptr<const Frame> frame1, int radius = 3);

        cv::Mat drawMat(const Image& mat);
 
        
     
}}}

#endif //VSLAM_LOG_H
