//
// Created by phil on 07.08.21.
//
#include <Eigen/Dense>
#include <opencv2/imgproc.hpp>
#include <opencv4/opencv2/core/eigen.hpp>
#include <opencv4/opencv2/highgui.hpp>

#include "Exceptions.h"
#include "core/Frame.h"
#include "core/Point3D.h"
#include "core/algorithm.h"
#include "visuals.h"
namespace pd {
namespace vision {
namespace vis {

cv::Mat drawFeatures(std::shared_ptr<const Frame> frame, int radius,
                     bool gradient) {
  cv::Mat mat;
  if (gradient) {
    cv::eigen2cv(frame->gradientImage(), mat);
  } else {
    cv::eigen2cv(frame->grayImage(), mat);
  }

  if (mat.cols == 0 || mat.rows == 0) {
    throw pd::Exception("Image is empty!");
  }

  for (const auto &ft : frame->features()) {
    cv::rectangle(
        mat,
        cv::Point(ft->position().x() - radius, ft->position().y() - radius),
        cv::Point(ft->position().x() + radius, ft->position().y() + radius),
        cv::Scalar(255, 255, 255));
  }

  return mat;
}

cv::Mat drawFeaturesWithPoints(std::shared_ptr<const Frame> frame, int radius) {

  cv::Mat mat;
  cv::eigen2cv(frame->gradientImage(), mat);

  if (mat.cols == 0 || mat.rows == 0) {
    throw pd::Exception("Image is empty!");
  }

  for (const auto &ft : frame->features()) {
    if (ft->point()) {
      cv::circle(mat, cv::Point(ft->position().x(), ft->position().y()), radius,
                 cv::Scalar(255, 255, 255));
    } else {
      cv::rectangle(
          mat,
          cv::Point(ft->position().x() - radius, ft->position().y() - radius),
          cv::Point(ft->position().x() + radius, ft->position().y() + radius),
          cv::Scalar(255, 255, 255));
    }
  }
  return mat;
}

cv::Mat drawFrame(std::shared_ptr<const Frame> frame) {

  cv::Mat mat;
  cv::eigen2cv(frame->grayImage(), mat);
  return mat;
}

/*       void Log::logFrame(std::shared_ptr<const FrameRGBD> frame, int level,
const std::string &name) { if (level <= _logLevel)
           {
               cv::Mat matRgb;
               cv::eigen2cv(frame->grayImage(),matRgb);
               const Eigen::MatrixXd matImage =
algorithm::normalize(frame->depthMap()) * 255.0; cv::Mat matD;
//                cv::eigen2cv(matImage.cast<std::uint8_t>(),matD);
//                const cv::Mat matRgbd = cv::hstack(matRgb,matD);
               logMat(matRgb, level, name);
           }

       }*/

cv::Mat drawReprojection(std::shared_ptr<const pd::vision::Frame> frame0,
                         std::shared_ptr<const pd::vision::Frame> frame1,
                         int radius) {

  cv::Mat mat;
  cv::eigen2cv(frame1->grayImage(), mat);

  if (mat.cols == 0 || mat.rows == 0) {
    throw pd::Exception("Image is empty!");
  }

  for (const auto &ft : frame0->features()) {
    if (ft->point()) {
      auto reprojectedPosition = frame1->world2image(ft->point()->position());
      cv::rectangle(mat,
                    cv::Point(reprojectedPosition.x() - radius,
                              reprojectedPosition.y() - radius),
                    cv::Point(reprojectedPosition.x() + radius,
                              reprojectedPosition.y() + radius),
                    cv::Scalar(255, 255, 255));
    }
  }

  return mat;
}


cv::Mat drawAsImage(const Eigen::MatrixXd& mat)
{
    return drawMat((algorithm::normalize(mat)*255).cast<uint8_t>());
}
cv::Mat drawMat(const Image &matEigen) {
  cv::Mat mat;
  cv::eigen2cv(matEigen, mat);

  return mat;
}

void Histogram::plot() const
{
    const double minH = _h.minCoeff();
    const double maxH = _h.maxCoeff();
    const double range = maxH - minH;
    const double binSize = range/(double)_nBins;
    std::vector<int> bins(_nBins,0);
    std::vector<std::string> ticksS(_nBins);
    std::vector<int> ticks(_nBins);
    for(int i = 0; i < _nBins; i++)
    {
       ticksS[i] = std::to_string(i * binSize + minH);
       ticks[i] = i;
    }
    for(int i = 0; i < _h.rows(); i++)
    {
        bins[ (int)((_h(i) - minH) / binSize)]++;
    }
    plt::figure();
    plt::title(_title.c_str());
    std::vector<double> hv(_h.data(),_h.data()+_h.rows());
    plt::hist(hv);
   // plt::xticks(ticks,ticksS);
}

void PlotLevenbergMarquardt::plot() const
{
    plt::figure();
    plt::subplot(1,5,1);
    plt::title("Squared Error $\\chi^2$");
    std::vector<double> chi2v(_chi2.data(), _chi2.data() + _nIterations);
    plt::named_plot("$\\chi^2$",chi2v);
    plt::xlabel("Iteration");
    plt::legend();

    plt::subplot(1,5,2);
    plt::title("Error Reduction $\\Delta \\chi^2$");
    std::vector<double> chi2predv(_chi2pred.data(), _chi2pred.data() + _nIterations);
    plt::named_plot("$\\Delta \\chi^2*$",chi2predv);
    std::vector<double> dChi2v(_dChi2.data(), _dChi2.data() + _nIterations);
    plt::named_plot("$\\Delta \\chi^2$",dChi2v);
    plt::xlabel("Iteration");
    plt::legend();

    plt::subplot(1,5,3);
    plt::title("Improvement Ratio $\\rho$");
    std::vector<double> rhov(_rho.data(), _rho.data() + _nIterations);
    plt::named_plot("$\\rho$",rhov);
    //plt::ylim(0.0,1.0);
    plt::xlabel("Iteration");
    plt::legend();

    plt::subplot(1,5,4);
    plt::title("Damping Factor $\\lambda$");
    std::vector<double> lambdav(_lambda.data(), _lambda.data() + _nIterations);
    plt::named_plot("$\\lambda$",lambdav);
    plt::xlabel("Iteration");

    plt::legend();
    plt::subplot(1,5,5);
    plt::title("Step Size $||\\Delta x||_2$");
    std::vector<double> stepsizev(_stepSize.data(), _stepSize.data() + _nIterations);
    plt::named_plot("$||\\Delta x||_2$",stepsizev);
    plt::xlabel("Iteration");
    plt::legend();
  

}
std::string PlotLevenbergMarquardt::csv() const 
{
    return "";
}

void PlotGaussNewton::plot() const
{
    plt::figure();
    plt::subplot(1,3,1);
    plt::title("Squared Error $\\chi^2$");
    std::vector<double> chi2v(_chi2.data(), _chi2.data() + _nIterations);
    plt::named_plot("$\\chi^2$",chi2v);
    plt::xlabel("Iteration");
    plt::legend();

    plt::subplot(1,3,2);
    plt::title("Error Reduction $\\Delta \\chi^2$");
    
    std::vector<double> dChi2v(_nIterations);
    dChi2v[0] = 0;
    for (int i = 1; i < _nIterations; i++)
    {
        dChi2v[i] = chi2v[i] - chi2v[i-1];
    }
    plt::named_plot("$\\Delta \\chi^2$",dChi2v);
    plt::xlabel("Iteration");
    plt::legend();

    plt::legend();
    plt::subplot(1,3,3);
    plt::title("Step Size $||\\Delta x||_2$");
    std::vector<double> stepsizev(_stepSize.data(), _stepSize.data() + _nIterations);
    plt::named_plot("$||\\Delta x||_2$",stepsizev);
    plt::xlabel("Iteration");
    plt::legend();
  

}
std::string PlotGaussNewton::csv() const 
{
    return "";
}

} // namespace vis
} // namespace vision
} // namespace pd
