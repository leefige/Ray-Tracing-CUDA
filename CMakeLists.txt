cmake_minimum_required (VERSION 3.10)

project(raytracing LANGUAGES CXX CUDA)

add_definitions(-DOLD_CXX)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  message(STATUS "Set up ccache ...")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT _IsIPOSupported)
  if(_IsIPOSupported)
  message(STATUS "Turn on INTERPROCEDURAL_OPTIMIZATION")
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp")

include_directories(
    raytracing
)

aux_source_directory(raytracing SRCS)

# require Threads
find_package(Threads REQUIRED)

add_executable(${PROJECT_NAME} ${SRCS})
target_link_libraries(raytracing Threads::Threads)
target_link_libraries(${PROJECT_NAME} stdc++fs)
