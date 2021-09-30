cmake_minimum_required(VERSION 3.8)
project(direct_image_alignmentTest)
include(../tools/Macros.cmake)

enable_testing()
find_package(GTest MODULE REQUIRED)

if(NOT TARGET pd::vslam)
	find_package(vslam REQUIRED)
endif()

pd_add_test(camera vslam)
pd_add_test(frame vslam)
pd_add_test(image_alignment vslam)
pd_add_test(image_alignment_dense vslam)

pd_add_test(algorithm vslam)
pd_add_test(feature_extraction vslam)
pd_add_test(stereo_alignment vslam)
pd_add_test(motion_prediction vslam)

target_compile_definitions(image_alignmentTest PUBLIC TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/resource")

target_compile_definitions(image_alignment_denseTest PUBLIC TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/resource")
