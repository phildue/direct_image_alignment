#include "Frame.h"
namespace pd::vision
{
        RgbdPyramid::RgbdPyramid(const Image intensity, const DepthMap& depth, Camera::ConstShPtr cam, uint32_t levels, const Timestamp& t, const PoseWithCovariance& pose)
        :_t(t)
        ,_pose(pose)
        {
                Mat<double,5,5> gaussianKernel;
                gaussianKernel << 1,4,6,4,1,
                                4,16,24,16,4,
                                6,24,36,24,6,
                                4,16,24,16,4,
                                1,4,6,4,1;
                                
                _levels.resize(levels);
                _scales.resize(levels);
                for(size_t i = 0; i < _levels.size(); i++)
                {
                        if(i == 0)
                        {
                                _levels[i] = std::make_shared<FrameRgbd>(intensity,depth,cam,t,pose);
                                _scales[i] = 1.0;
                        }else{
                                const double s = 0.5;
                                _scales[i] = _scales[i-1]/s;
                                Image imgBlur = algorithm::conv2d(this->intensity(i-1).cast<double>(),gaussianKernel).cast<uint8_t>();
                                imgBlur.col(0) = imgBlur.col(2);
                                imgBlur.col(1) = imgBlur.col(2);
                                imgBlur.col(imgBlur.cols()-2) = imgBlur.col(imgBlur.cols()-3);
                                imgBlur.col(imgBlur.cols()-1) = imgBlur.col(imgBlur.cols()-3);
                                imgBlur.row(0) = imgBlur.row(2);
                                imgBlur.row(1) = imgBlur.row(2);
                                imgBlur.row(imgBlur.rows()-2) = imgBlur.row(imgBlur.rows()-3);
                                imgBlur.row(imgBlur.rows()-1) = imgBlur.row(imgBlur.rows()-3);

                                Image imgRes = algorithm::resize(imgBlur,s);
                                DepthMap depthBlur = algorithm::medianBlur<double>(this->depth(i-1),3,3);
                                DepthMap depthRes = algorithm::resize(depthBlur,s);

                                _levels[i] = std::make_shared<FrameRgbd>(
                                        imgRes,
                                        depthRes,
                                        Camera::resize(this->camera(i-1),s),
                                        t,
                                        pose);

                        } 
                }
        }
        RgbdPyramid::RgbdPyramid(const Image intensity, const DepthMap& depth, Camera::ConstShPtr cam,const std::vector<double>& scales, const Timestamp& t, const PoseWithCovariance& pose)
        :RgbdPyramid(intensity,depth,cam,scales.size(),t,pose){}

} // namespace pd::vision


