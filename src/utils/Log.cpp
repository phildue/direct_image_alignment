//
// Created by phil on 07.08.21.
//

#include "Log.h"

void Log::init(int loglevel) {
    FLAGS_logtostderr = true;
    FLAGS_v = loglevel;
}
