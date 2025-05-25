// src/compute/boundary.hpp

#pragma once

#include <boost/multiprecision/cpp_dec_float.hpp>
#include <future>
#include <utility>
#include "settings.hpp"

using BigFloat = boost::multiprecision::cpp_dec_float_50;

class BoundaryService {
public:
    BoundaryService();
    void update(const Settings& settings);
    std::pair<BigFloat,BigFloat> getNextTarget() const;

private:
    std::future<std::pair<float,float>> task_;
    BigFloat nextX_, nextY_;
    double   lastRecalcTime_ = 0.0;
};
