//
// Created by phil on 17.08.21.
//

#ifndef VSLAM_DESCRIPTOR_H
#define VSLAM_DESCRIPTOR_H

#include <Eigen/Dense>
namespace pd { namespace vision{
    class Descriptor {
public:
    virtual Eigen::MatrixXd mat() const = 0;
};

class GradientDescriptor : public Descriptor
{
public:
    GradientDescriptor(double magnitude)
    :_magnitude(magnitude)
    {}

    Eigen::MatrixXd mat() const override {return Eigen::Matrix<double,1,1>(_magnitude);};

private:
    int _magnitude;
};
}}

#endif //VSLAM_DESCRIPTOR_H
