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

        cv::Mat drawAsImage(const Eigen::MatrixXd& mat);
        //static void logJacobianImage(int iteration, const Eigen::MatrixXd& jacobian);

        cv::Mat drawFeaturesWithPoints(std::shared_ptr<const Frame> frame, int radius = 3);
        cv::Mat drawReprojection(std::shared_ptr<const Frame> frame0,std::shared_ptr<const Frame> frame1, int radius = 3);

        cv::Mat drawMat(const Image& mat);
 
        class Drawable{
            public:
            typedef std::unique_ptr<Drawable> Ptr;
            typedef std::unique_ptr<const Drawable> ConstPtr;
            typedef std::shared_ptr<Drawable> ShPtr;
            typedef std::shared_ptr<const Drawable> ConstShPtr;
          
            virtual cv::Mat draw() const = 0;
        };
        template <typename T>
        class DrawableMat : public Drawable{
            public:
            DrawableMat(const Eigen::Matrix<T,-1,-1>& mat):_mat(mat){}
            cv::Mat draw() const override { return drawAsImage(_mat.template cast<double>());}
            private:
            const Eigen::Matrix<T,-1,-1> _mat;
        };

        class Plot{
            public:
            typedef std::unique_ptr<Plot> Ptr;
            typedef std::unique_ptr<const Plot> ConstPtr;
            typedef std::shared_ptr<Plot> ShPtr;
            typedef std::shared_ptr<const Plot> ConstShPtr;
          
           
            virtual void plot() const = 0;
            virtual std::string csv() const = 0;
        };

        
        class PlotLevenbergMarquardt : public Plot{

            public:
            PlotLevenbergMarquardt(const Eigen::VectorXd& chi2,const Eigen::VectorXd &chi2pred,const Eigen::VectorXd& lambda,const Eigen::VectorXd& stepSize)
            : _chi2(chi2)
            , _chi2pred(chi2pred)
            , _lambda(lambda)
            , _stepSize(stepSize)
            {}

            void plot() const override;
            std::string csv() const override;
            private:
            const Eigen::VectorXd _chi2;
            const Eigen::VectorXd _chi2pred;
            const Eigen::VectorXd _lambda;
            const Eigen::VectorXd _stepSize;

        };
     
}}}

#endif //VSLAM_LOG_H
