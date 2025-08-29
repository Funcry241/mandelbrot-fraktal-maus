#pragma once
// Host-only: niemals in .cu inkludieren!
#include <boost/multiprecision/cpp_dec_float.hpp>
namespace nm {
  using real = boost::multiprecision::cpp_dec_float_100; // 100 Dezimalstellen (später anhebbar)
  struct cplx { real re, im; };
}
