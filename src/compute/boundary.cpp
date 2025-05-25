// src/compute/boundary.cpp

#include "compute/boundary.hpp"
#include "mandelbrot.hpp"
#include <GLFW/glfw3.h>
#include <cmath>
#include <chrono>

extern Settings S;

BoundaryService::BoundaryService()
 : nextX_(S.offsetX), nextY_(S.offsetY),
   lastRecalcTime_(glfwGetTime())
{
    double z  = S.zoom.convert_to<double>();
    double ox = S.offsetX.convert_to<double>();
    double oy = S.offsetY.convert_to<double>();
    task_ = std::async(std::launch::async,
                       computeBoundaryGPU,
                       z, ox, oy,
                       S.width, S.height,
                       S.sampleStep, S.maxIter);
}

void BoundaryService::update(const Settings& settings) {
    if (task_.valid() &&
        task_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        auto [fx,fy] = task_.get();
        nextX_ = BigFloat(fx);
        nextY_ = BigFloat(fy);
        lastRecalcTime_ = glfwGetTime();
        double z  = settings.zoom.convert_to<double>();
        double ox = settings.offsetX.convert_to<double>();
        double oy = settings.offsetY.convert_to<double>();
        task_ = std::async(std::launch::async,
                           computeBoundaryGPU,
                           z, ox, oy,
                           settings.width, settings.height,
                           settings.sampleStep, settings.maxIter);
    }
}

std::pair<BigFloat,BigFloat> BoundaryService::getNextTarget() const {
    return { nextX_, nextY_ };
}
