cmake_minimum_required(VERSION 3.9)
project(mybt)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# 设置头文件&相关库
find_package(BLAS REQUIRED)
find_package(Eigen3 REQUIRED CONFIG) 
find_package(pybind11 REQUIRED)


# 模块
pybind11_add_module(mybt backtest.cpp)