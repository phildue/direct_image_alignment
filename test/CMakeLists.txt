cmake_minimum_required(VERSION 3.8)
project(direct_image_alignmentTest)
include(../tools/Macros.cmake)

enable_testing()
find_package(GTest CONFIG REQUIRED)

if(NOT TARGET pd::vslam)
	find_package(vslam REQUIRED)
endif()

pd_add_test(camera vslam)
pd_add_test(frame vslam)

pd_add_test(algorithm vslam)
pd_add_test(feature_extraction vslam)
pd_add_test(stereo_alignment vslam)
pd_add_test(motion_prediction vslam)
#pd_add_test(solver vslam)
pd_add_test(lukas_kanade_flow vslam)
pd_add_test(lukas_kanade_affine vslam)
pd_add_test(lukas_kanade_se3 vslam)
pd_add_test(log vslam)
pd_add_test(plot vslam)
pd_add_test(loss vslam)

#target_compile_definitions(image_alignmentTest PUBLIC TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/resource")

#target_compile_definitions(solverTest PUBLIC TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/resource")
target_compile_definitions(lukas_kanade_affineTest PUBLIC TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/resource")
target_compile_definitions(lukas_kanade_flowTest PUBLIC TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/resource")
target_compile_definitions(lukas_kanade_se3Test PUBLIC TEST_RESOURCE="${CMAKE_CURRENT_LIST_DIR}/resource")
