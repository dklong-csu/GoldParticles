cmake_minimum_required(VERSION 3.1...3.27)

# Name of project
project(
    DQMOM
    VERSION 1.0
    LANGUAGES CXX
)

# Set CMAKE_ variables
set (CMAKE_CXX_STANDARD 20)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_EXTENSIONS OFF)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

# Find packages
FIND_PACKAGE (
        _sundials
        NAMES SUNDIALS
        HINTS ${SUNDIALS_DIR} 
        REQUIRED
)
IF ("${_sundials}" STREQUAL "_sundials-NOTFOUND")
  MESSAGE(FATAL_ERROR
          "The SUNDIALS library was not found. You have to specify a path "
          "to that library by setting the SUNDIALS_DIR environment variable, "
          "or by passing '-DSUNDIALS_DIR=...' as an argument to 'cmake'.")
ELSE()
  MESSAGE(STATUS "Found SUNDIALS")
ENDIF()


FIND_PATH(_sundials_include_dir
        NAMES cvode/cvode.h
        HINTS ${SUNDIALS_DIR}/include)
IF ("${_sundials_include_dir}" STREQUAL "_sundials_include_dir-NOTFOUND")
  MESSAGE(FATAL_ERROR
          "The SUNDIALS headers were not found. You have to specify a path "
          "to that library by setting the SUNDIALS_DIR environment variable, "
          "or by passing '-DSUNDIALS_DIR=...' as an argument to 'cmake'.")
ELSE()
  MESSAGE(STATUS "Found SUNDIALS headers at ${_sundials_include_dir}")
  INCLUDE_DIRECTORIES(${_sundials_include_dir})
ENDIF()


FIND_LIBRARY(_sundials_cvode_lib
        NAMES sundials_cvode
        HINTS ${SUNDIALS_DIR}/lib)
IF ("${_sundials_cvode_lib}" STREQUAL "_sundials_cvode_lib-NOTFOUND")
  MESSAGE(FATAL_ERROR
          "The SUNDIALS CVODE library was not found. You have to specify a path "
          "to that library by setting the SUNDIALS_DIR environment variable, "
          "or by passing '-DSUNDIALS_DIR=...' as an argument to 'cmake'.")
ELSE()
  MESSAGE(STATUS "Found SUNDIALS CVODE library at ${_sundials_cvode_lib}")
  LINK_DIRECTORIES(${SUNDIALS_DIR}/lib)
ENDIF()



find_package(MUQ REQUIRED)
include_directories(${MUQ_INCLUDE_DIRS})

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

# Define the library
# add_library(myLib
#     user_data.hpp)

# Executable
add_executable(dqmom dqmom.cpp)

# Link executable(s) to library
target_link_libraries(dqmom PUBLIC SUNDIALS::cvode)
target_link_libraries(dqmom PUBLIC ${MUQ_LIBRARIES} ${MUQ_LINK_LIBRARIES})
target_link_libraries(dqmom PUBLIC ${Boost_LIBRARIES})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(dqmom PUBLIC OpenMP::OpenMP_CXX)
endif()
