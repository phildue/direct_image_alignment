#include "Frame.h"
namespace pd::vision
{
        FrameRgb::FrameRgb(const Image& intensity, Camera::ConstShPtr cam, size_t nLevels, const Timestamp& t, const PoseWithCovariance& pose)
        :_t(t),
        _pose(pose){

                _intensity.resize(nLevels);
                _dIx.resize(nLevels);
                _dIy.resize(nLevels);
                _cam.resize(nLevels);

                //TODO make based on scales
                Mat<double,5,5> gaussianKernel;
                gaussianKernel << 1,4,6,4,1,
                                4,16,24,16,4,
                                6,24,36,24,6,
                                4,16,24,16,4,
                                1,4,6,4,1;
                const double s = 0.5;
                for(size_t i = 0; i < nLevels; i++)
                {
                        if(i == 0)
                        {
                                _intensity[i] = intensity;
                                _cam[i] = cam;
                                
                        }else{
                                Image imgBlur = algorithm::conv2d(_intensity[i-1].cast<double>(),gaussianKernel).cast<uint8_t>();
                                //TODO move padding to separate function
                                imgBlur.col(0) = imgBlur.col(2);
                                imgBlur.col(1) = imgBlur.col(2);
                                imgBlur.col(imgBlur.cols()-2) = imgBlur.col(imgBlur.cols()-3);
                                imgBlur.col(imgBlur.cols()-1) = imgBlur.col(imgBlur.cols()-3);
                                imgBlur.row(0) = imgBlur.row(2);
                                imgBlur.row(1) = imgBlur.row(2);
                                imgBlur.row(imgBlur.rows()-2) = imgBlur.row(imgBlur.rows()-3);
                                imgBlur.row(imgBlur.rows()-1) = imgBlur.row(imgBlur.rows()-3);

                                _intensity[i] = algorithm::resize(imgBlur,s);
                                _cam[i] = Camera::resize(_cam[i-1],s);

                        } 
                        _dIx[i] = algorithm::conv2d(_intensity[i].cast<double>(),Kernel2d<double>::sobelX()).cast<int>();
                        _dIy[i] = algorithm::conv2d(_intensity[i].cast<double>(),Kernel2d<double>::sobelY()).cast<int>();
                                
                }

        }


        FrameRgbd::FrameRgbd(const Image& intensity,const MatXd& depth, Camera::ConstShPtr cam, size_t nLevels, const Timestamp& t, const PoseWithCovariance& pose)
        :FrameRgb(intensity,cam,nLevels,t,pose)
        {
                auto depth2pcl = [](const DepthMap& d, Camera::ConstShPtr c){
                                std::vector<Vec3d> pcl(d.rows()*d.cols());
                                for(int v = 0; v < d.rows(); v++)
                                {
                                        for(int u = 0; u < d.cols(); u++)
                                        {
                                                /* Exclude pixels that are close to not having depth since we do bilinear interpolation later*/
                                                if (std::isfinite(d(v,u)) && d(v,u) > 0 &&
                                                std::isfinite(d(v+1,u+1)) && d(v+1,u+1) > 0  &&
                                                std::isfinite(d(v+1,u-1)) && d(v+1,u-1) > 0  &&
                                                std::isfinite(d(v-1,u+1)) && d(v-1,u+1) > 0  &&
                                                std::isfinite(d(v-1,u-1)) && d(v-1,u-1) > 0
                                                )//TODO move to actual interpolation?
                                                {
                                                        pcl[v * d.cols() + u] = c->image2camera({u,v},d(v,u));
                                                }else{
                                                        pcl[v * d.cols() + u] = Eigen::Vector3d::Zero();
                                                }
                                        }
                                }
                                return pcl;
                        };
                _depth.resize(nLevels);
                _pcl.resize(nLevels);
                const double s = 0.5;
                for(size_t i = 0; i < nLevels; i++)
                {
                        if(i == 0)
                        {
                                _depth[i] = depth;
                                _pcl[i] = depth2pcl(depth,cam);
                                
                        }else{
                                DepthMap depthBlur = algorithm::medianBlur(_depth[i-1],3,3);
                                _depth[i] = algorithm::resize(depthBlur,s);
                                _pcl[i] = depth2pcl(depth,Camera::resize(cam,s));

                        } 
                                
                }
        }
        std::vector<Vec3d> FrameRgbd::pcl(size_t level, bool removeInvalid) const
        {
                if(removeInvalid)
                {
                        std::vector<Vec3d> pcl;
                        pcl.reserve(_pcl.at(level).size());
                        std::copy_if(_pcl.at(level).begin(),_pcl.at(level).end(),std::back_inserter(pcl),[](auto p){return p.z() <= 0 || !std::isfinite(p.z());});
                        return pcl;
                }else{
                        return _pcl.at(level);
                }
        }


} // namespace pd::vision


