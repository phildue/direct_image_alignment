//
// Created by phil on 10.10.20.
//

#include <gtest/gtest.h>
#include "utils/visuals.h"
#include "utils/Exceptions.h"
using namespace testing;
using namespace pd;
using namespace pd::vision;
#define VISUALIZE true
TEST(LogTest,Plot)
{

    vis::plt::plot({1,3,2,4});
    vis::plt::figure();
    vis::plt::hist<double>({1.0,3.0,2.0,4.0});

    if(VISUALIZE)
    {
        vis::plt::show();
    }
}

TEST(LogTest,Histogram)
{
    Eigen::Vector<double,9> v;
    v << -1,-1,1,3,3,3,0,0,0;
    auto h = std::make_shared<vis::Histogram>(v,"TestHistogram");
    h->plot();
    if(VISUALIZE)
    {
        vis::plt::show();
    }
}

TEST(LogTest,HistogramLarge)
{
    Eigen::VectorXd v(20000);
    for (int i = -5000; i < 15000; i++)
    {
        v(i+5000) = i;
    }
    auto h = std::make_shared<vis::Histogram>(v,"TestHistogram");
    h->plot();
    if(VISUALIZE)
    {
        vis::plt::show();
    }
}