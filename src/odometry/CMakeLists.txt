cmake_minimum_required(VERSION 3.8)
project(odometry VERSION 1.0.0)

set(sources
        src/RgbdOdometry.cpp
        src/RgbdOdometryKalman.cpp
        )

set(headers
        src/RgbdOdometry.h
        src/RgbdOdometryKalman.h
        include/odometry/odometry.h
        )

pd_setup_lib(${PROJECT_NAME} ${PROJECT_VERSION} "${sources}" "${headers}" pd)

target_link_libraries(${PROJECT_NAME} kalman lukas_kanade utils core)

target_link_libraries(${PROJECT_NAME} core utils solver)
if (UNIX)
	target_link_libraries(${PROJECT_NAME} stdc++fs)
endif()

find_package(Eigen3 3.4 REQUIRED)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen3)

find_package(Sophus REQUIRED)
target_link_libraries(${PROJECT_NAME} Sophus::Sophus)


# Add tests
option(VSLAM_BUILD_TESTS "Enable to build tests" ON)
if(VSLAM_BUILD_TESTS)
	enable_testing()
	add_subdirectory(test)
endif()
