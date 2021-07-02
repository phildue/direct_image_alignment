cmake_minimum_required(VERSION 3.8)
project(direct_image_alignment VERSION 1.0.0)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/tools
		${CMAKE_BINARY_DIR}/)
include(Macros)

set(sources
		src/Frame.cpp
		src/ImageAlignment.cpp
		src/Feature2D.cpp
		src/Point3D.cpp
		src/Pose.cpp
		src/Camera.cpp
		src/math.cpp
		)

set(headers
		src/Frame.h
		src/ImageAlignment.h
		src/Feature2D.h
		src/Point3D.h
		src/Pose.h
		src/Camera.h
		src/math.h
		include/direct_image_alignment/direct_image_alignment.h
		)

setup_package(${PROJECT_NAME} ${PROJECT_VERSION} "${sources}" "${headers}" PD)


# Dependencies
include(conanbuildinfo)
conan_basic_setup(TARGETS)

target_link_libraries(${PROJECT_NAME} CONAN_PKG::Eigen3)
target_link_libraries(${PROJECT_NAME} CONAN_PKG::ceres-solver)
target_link_libraries(${PROJECT_NAME} CONAN_PKG::sophus)


# Add tests
option(MyLibrary_BUILD_TESTS "Enable to build tests" ON)
if(MyLibrary_BUILD_TESTS)
	enable_testing()
	add_subdirectory(test)
endif()
