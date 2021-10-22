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
		src/solver/LevenbergMarquardt.cpp
		src/solver/GaussNewton.cpp
		src/image_alignment/ImageAlignmentSparse.cpp
		src/image_alignment/ImageAlignmentDense.cpp
		src/utils/visuals.cpp
		src/utils/Log.cpp
		src/utils/utils.cpp
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
		src/solver/GaussNewton.h
		src/feature_extraction/Descriptor.h
		src/utils/Exceptions.h
		src/utils/visuals.h
		src/utils/Log.h
		src/utils/utils.h
		src/image_alignment/SE3Parameterization.h
		src/image_alignment/ImageAlignmentSparseCeresAutoDiff.hpp
		src/image_alignment/ImageAlignmentSparseCeres.hpp
		src/image_alignment/ImageAlignmentSparse.h
		src/image_alignment/ImageAlignmentDense.h
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
$<BUILD_INTERFACE:CFG_DIR="${CMAKE_CURRENT_LIST_DIR}/cfg">
$<INSTALL_INTERFACE:CFG_DIR="cfg">)

target_compile_definitions(${PROJECT_NAME} PUBLIC 
$<BUILD_INTERFACE:LOG_DIR="${CMAKE_BINARY_DIR}/log">
$<INSTALL_INTERFACE:CFG_DIR="/tmp/vslam/log">)

# Dependencies

if (UNIX)
	target_link_libraries(${PROJECT_NAME} stdc++fs)
endif()

include(conanbuildinfo)
conan_basic_setup(TARGETS)

#Cannot get ceres via conan since package is broken
find_package(Ceres REQUIRED)
target_link_libraries(${PROJECT_NAME} Ceres::ceres)
#target_link_libraries(${PROJECT_NAME} CONAN_PKG::ceres-solver)


	target_link_libraries(${PROJECT_NAME} CONAN_PKG::eigen)

if (DEFINED OpenCV_LIBRARIES)
	target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
else()
	target_link_libraries(${PROJECT_NAME} CONAN_PKG::opencv)
endif()

target_link_libraries(${PROJECT_NAME} CONAN_PKG::sophus)
target_link_libraries(${PROJECT_NAME} CONAN_PKG::glog)
target_link_libraries(${PROJECT_NAME} CONAN_PKG::easyloggingpp)


# Add tests
option(MyLibrary_BUILD_TESTS "Enable to build tests" ON)
if(MyLibrary_BUILD_TESTS)
	enable_testing()
	add_subdirectory(test)
endif()
