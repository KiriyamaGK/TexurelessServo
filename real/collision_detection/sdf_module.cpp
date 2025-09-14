#include <vector>
#include <array>
#include <memory>
#include "TriangleMeshDistance.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace tmd;  // 添加这行使用tmd命名空间

class SDFCalculator {
public:
    void load_mesh(const std::vector<std::array<double, 3>>& vertices,
                 const std::vector<std::array<int, 3>>& triangles) {
        mesh_distance.reset(new TriangleMeshDistance(vertices, triangles));  // 修改为直接new
    }
    
    double signed_distance(const std::array<double, 3>& point) {
        auto result = mesh_distance->signed_distance(point);
        return result.distance;
    }
    
    bool is_colliding(const std::array<double, 3>& point, double threshold = 0.0) {
        auto result = mesh_distance->signed_distance(point);
        return result.distance < threshold;
    }

private:
    std::unique_ptr<TriangleMeshDistance> mesh_distance;  // 现在可以正确识别类型
};

PYBIND11_MODULE(sdf_module, m) {
    py::class_<SDFCalculator>(m, "SDFCalculator")
        .def(py::init<>())
        .def("load_mesh", &SDFCalculator::load_mesh)
        .def("signed_distance", &SDFCalculator::signed_distance)
        .def("is_colliding", &SDFCalculator::is_colliding);
}