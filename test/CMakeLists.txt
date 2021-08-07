cmake_minimum_required(VERSION 3.8)
project(direct_image_alignmentTest)
include(../tools/Macros.cmake)

enable_testing()
find_package(GTest MODULE REQUIRED)

if(NOT TARGET PD::direct_image_alignment)
	find_package(direct_image_alignment REQUIRED)
endif()


pd_add_test(camera direct_image_alignment)
pd_add_test(frame direct_image_alignment)
pd_add_test(image_alignment direct_image_alignment)
pd_add_test(math direct_image_alignment)

