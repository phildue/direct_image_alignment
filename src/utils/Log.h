//
// Created by phil on 07.08.21.
//

#ifndef VSLAM_LOG_H
#define VSLAM_LOG_H

#include <glog/logging.h>

class Log {
public:
    static void init(int loglevel = 3);
};


#endif //VSLAM_LOG_H
