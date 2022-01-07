cmake_minimum_required(VERSION 3.8)
project(vslam VERSION 1.0.0)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/tools
		${CMAKE_BINARY_DIR}/)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})

include(Macros)
set(sources
		src/core/Frame.cpp
		src/core/Feature2D.cpp
		src/core/Point3D.cpp
                src/core/Camera.cpp
		src/core/algorithm.cpp
		src/solver/Loss.cpp
		src/utils/visuals.cpp
		src/utils/Log.cpp
		src/utils/utils.cpp
		src/odometry/odometry.cpp
		src/lukas_kanade/WarpOpticalFlow.cpp
		src/lukas_kanade/WarpAffine.cpp
		src/lukas_kanade/WarpSE3.cpp
                src/system/StereoAlignment.cpp
		src/feature_extraction/FeatureExtraction.cpp
		src/feature_extraction/FeatureExtractionOpenCv.cpp
		src/motion_prediction/MotionPrediction.cpp
		${CMAKE_BINARY_DIR}/include/easylogging++.cc
		)

set(headers
		src/core/Frame.h
		src/core/Feature2D.h
		src/core/Point3D.h
                src/core/Camera.h
		src/core/algorithm.h
		src/solver/LevenbergMarquardt.h
		src/solver/LevenbergMarquardt.hpp
		src/solver/GaussNewton.hpp
		src/solver/GaussNewton.h
		src/solver/Loss.h
		src/lukas_kanade/LukasKanade.hpp
		src/lukas_kanade/LukasKanade.h
		src/lukas_kanade/LukasKanadeInverseCompositional.hpp
		src/lukas_kanade/LukasKanadeInverseCompositional.h
		src/odometry/odometry.h
		src/lukas_kanade/WarpOpticalFlow.h
		src/lukas_kanade/WarpAffine.h
		src/lukas_kanade/WarpSE3.h
		src/solver/solver.h
		src/feature_extraction/Descriptor.h
		src/utils/Exceptions.h
		src/utils/visuals.h
		src/utils/Log.h
		src/utils/utils.h
		src/feature_extraction/FeatureExtraction.h
		src/feature_extraction/FeatureExtractionOpenCv.h
		src/feature_extraction/feature_extraction.h
                src/system/StereoAlignment.h
		src/motion_prediction/MotionPrediction.h
		include/vslam/vslam.h
		)

pd_setup_lib(${PROJECT_NAME} ${PROJECT_VERSION} "${sources}" "${headers}" pd)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_compile_definitions(${PROJECT_NAME} PUBLIC 
$<BUILD_INTERFACE:CFG_DIR="${CMAKE_CURRENT_LIST_DIR}/cfg/">
$<INSTALL_INTERFACE:CFG_DIR="/share/${PROJECT_NAME}/cfg/">)

target_compile_definitions(${PROJECT_NAME} PUBLIC 
$<BUILD_INTERFACE:LOG_DIR="${CMAKE_BINARY_DIR}/log">
$<INSTALL_INTERFACE:CFG_DIR="/tmp/vslam/log">)

target_compile_definitions(${PROJECT_NAME} PUBLIC
ELPP_FEATURE_PERFORMANCE_TRACKING
)
option(VSLAM_PERFORMANCE_TRACKING Off)
if(NOT VSLAM_PERFORMANCE_TRACKING)
	target_compile_definitions(${PROJECT_NAME} PUBLIC
		ELPP_DISABLE_PERFORMANCE_TRACKING)	
endif()
# Dependencies

if (UNIX)
	target_link_libraries(${PROJECT_NAME} stdc++fs)
endif()

#Cannot get ceres via conan since package is broken
#find_package(Ceres REQUIRED)
#target_link_libraries(${PROJECT_NAME} Ceres::ceres)
#target_link_libraries(${PROJECT_NAME} CONAN_PKG::ceres-solver)


# Try for OpenCV 4.X, but settle for whatever is installed
find_package(OpenCV 4 QUIET)
if (NOT OpenCV_FOUND)
  find_package(OpenCV REQUIRED)
endif ()
message(STATUS "Found OpenCV version ${OpenCV_VERSION}")
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
find_package(Eigen3 3.4 REQUIRED)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen3)

find_package(Sophus REQUIRED)
find_package(easyloggingpp REQUIRED)
target_link_libraries(${PROJECT_NAME} Sophus::Sophus)
target_link_libraries(${PROJECT_NAME} easyloggingpp::easyloggingpp)

find_package(matplotlib_cpp)
target_link_libraries(${PROJECT_NAME} matplotlib_cpp::matplotlib_cpp)

# Add tests
option(VSLAM_BUILD_TESTS "Enable to build tests" ON)
if(VSLAM_BUILD_TESTS)
	enable_testing()
	add_subdirectory(test)
endif()
