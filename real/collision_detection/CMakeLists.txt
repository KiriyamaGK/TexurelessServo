cmake_minimum_required(VERSION 3.12)
project(sdf_module)

find_package(pybind11 REQUIRED)

pybind11_add_module(sdf_module sdf_module.cpp)