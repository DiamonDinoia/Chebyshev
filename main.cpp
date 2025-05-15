#include <iostream>
#include <random>
#include <cmath>
#include <vector>
#include <array>
#include <numeric>
#include <algorithm>

#include <xtensor/containers/xarray.hpp>
#include <xtensor/optional/xoptional_assembly.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor-blas/xlinalg.hpp>

constexpr double pi = 3.14159265358979323846;

//==============================================================================
// ChebFunND: N dims → M outputs, all in xtensor_fixed
//==============================================================================
template <typename T, size_t N, size_t M, typename F, int... Degrees>
class ChebFunND {
  static_assert(sizeof...(Degrees) == N, "Degree pack must match dimensionality");

  template <int... Ds>
  static constexpr auto make_coeffs_shape() {
    return xt::xshape<(Ds + 1)..., M>();
  }

  using coeffs_t = xt::xtensor_fixed<T, decltype(make_coeffs_shape<Degrees...>())>;
  using vecN_t   = xt::xtensor_fixed<T, xt::xshape<N>>;
  using vecM_t   = xt::xtensor_fixed<T, xt::xshape<M>>;

  coeffs_t              coeffs_;
  vecN_t                a_, b_;
  std::array<size_t, N> shape_;

public:
  ChebFunND(const vecN_t &a, const vecN_t &b, F f)
    : a_(a), b_(b)
  {
    // 1) Record each dimension’s size
    for (size_t d = 0; d < N; ++d) {
      shape_[d] = coeffs_.shape()[d];
    }

    // 2) Sample f at Chebyshev nodes into coeffs_
    std::array<size_t, N> idx{};
    sample<0>(f, idx);

    // 3) Compute total coeffs and strides
    size_t total = 1;
    for (size_t d = 0; d < N; ++d) total *= shape_[d];
    std::array<size_t, N> strides{};
    strides[N-1] = 1;
    for (size_t i = N; i-- > 1; ) {
      strides[i-1] = strides[i] * shape_[i];
    }

    // 4) Prepare dynamic_view slice selector
    xt::xdynamic_slice_vector slice_selector(N+1);
    for (size_t d = 0; d < N; ++d) {
      slice_selector[d] = xt::all();
    }

    // 5) For each output component: gather, DCT, write back
    for (size_t o = 0; o < M; ++o) {
      slice_selector[N] = static_cast<long>(o);
      auto coeffs_slice = xt::dynamic_view(coeffs_, slice_selector);

      // gather
      std::vector<T> comp;
      comp.reserve(total);
      for (auto it = coeffs_slice.cbegin(); it != coeffs_slice.cend(); ++it) {
        comp.push_back(*it);
      }

      // separable N-D DCT on comp
      for (size_t axis = 0; axis < N; ++axis) {
        size_t m = shape_[axis];
        size_t axis_stride = strides[axis];
        size_t block = axis_stride * m;
        size_t outer = total / block;
        std::vector<T> buf(m), dct(m);
        for (size_t b = 0; b < outer; ++b) {
          size_t base = b * block;
          for (size_t off = 0; off < axis_stride; ++off) {
            // gather 1-D slice
            for (size_t j = 0; j < m; ++j) {
              buf[j] = comp[base + off + j * axis_stride];
            }
            // DCT
            for (size_t k = 0; k < m; ++k) {
              T sum{};
              for (size_t j = 0; j < m; ++j) {
                sum += buf[j] * std::cos(pi * k * (j + T(0.5)) / m);
              }
              dct[k] = sum * (T(2) / m);
            }
            dct[0] *= T(0.5);
            // scatter back
            for (size_t j = 0; j < m; ++j) {
              comp[base + off + j * axis_stride] = dct[j];
            }
          }
        }
      }

      // write back
      size_t k = 0;
      for (auto it = coeffs_slice.begin(); it != coeffs_slice.end(); ++it) {
        *it = comp[k++];
      }
    }
  }

  vecM_t operator()(const vecN_t &x) const {
    vecN_t xt = (T(2) * x - a_ - b_) / (b_ - a_);
    std::vector<std::vector<T>> Tvals(N);
    for (size_t d = 0; d < N; ++d) {
      size_t m = shape_[d];
      auto &Tv = Tvals[d]; Tv.resize(m);
      if (m > 0) Tv[0] = T(1);
      if (m > 1) Tv[1] = xt(d);
      for (size_t k = 2; k < m; ++k) {
        Tv[k] = T(2) * xt(d) * Tv[k-1] - Tv[k-2];
      }
    }
    vecM_t result = xt::zeros<T>({M});
    std::array<size_t, N> idx{};
    eval<0>(idx, T(1), Tvals, result);
    return result;
  }

private:
  template <size_t D_idx>
  void sample(F &f, std::array<size_t, N> &idx) {
    for (size_t i = 0; i < shape_[D_idx]; ++i) {
      idx[D_idx] = i;
      if constexpr (D_idx + 1 < N) {
        sample<D_idx+1>(f, idx);
      } else {
        vecN_t pt;
        for (size_t d = 0; d < N; ++d) {
          T θ = pi * (idx[d] + T(0.5)) / shape_[d];
          pt(d) = T(0.5)*(a_(d)+b_(d)) + T(0.5)*(b_(d)-a_(d))*std::cos(θ);
        }
        auto out = f(pt);
        for (size_t o = 0; o < M; ++o) {
          std::array<size_t, N+1> full;
          for (size_t d = 0; d < N; ++d) full[d] = idx[d];
          full[N] = o;
          coeffs_[full] = out(o);
        }
      }
    }
  }

  template <size_t D_idx>
  void eval(std::array<size_t, N> &idx, T prod,
            const std::vector<std::vector<T>> &Tvals,
            vecM_t &acc) const
  {
    if constexpr (D_idx == N) {
      for (size_t o = 0; o < M; ++o) {
        std::array<size_t, N+1> full;
        for (size_t d = 0; d < N; ++d) full[d] = idx[d];
        full[N] = o;
        acc(o) += coeffs_[full] * prod;
      }
    } else {
      for (size_t k = 0; k < shape_[D_idx]; ++k) {
        idx[D_idx] = k;
        eval<D_idx+1>(idx, prod * Tvals[D_idx][k], Tvals, acc);
      }
    }
  }
};

//==============================================================================
// Test harness
//==============================================================================
template <size_t D, size_t M, typename F>
void test_interp(const xt::xtensor_fixed<double, xt::xshape<D>> &a,
                 const xt::xtensor_fixed<double, xt::xshape<D>> &b,
                 F fcn)
{
  auto run_test = [&](auto &cheb){
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(0,1);
    xt::xtensor_fixed<double, xt::xshape<D>> pt;
    std::array<double, M> max_rel{};
    for (int i=0; i<10000; ++i) {
      for (size_t d=0; d<D; ++d) {
        double t = dist(rng);
        pt(d) = a(d) + t*(b(d)-a(d));
      }
      auto approx = cheb(pt);
      double s = xt::sum(pt)();
      xt::xtensor_fixed<double, xt::xshape<M>> exact;
      for (size_t k=0; k<M; ++k) exact(k) = (k+1.0)*std::exp(s);
      for (size_t k=0; k<M; ++k) {
        double e=exact(k), p=approx(k);
        double rel = std::abs(e)>1e-12 ? std::abs(1-p/e) : std::abs(p);
        max_rel[k] = std::max(max_rel[k], rel);
      }
    }
    std::cout<<D<<"-D -> "<<M<<" outputs, max rel errors:";
    for(double e: max_rel) std::cout<<' '<<e;
    std::cout<<"\n";
  };

  if constexpr (D==1) { ChebFunND<double,1,M,F,4>   cheb(a,b,fcn); run_test(cheb);}
  if constexpr (D==2) { ChebFunND<double,2,M,F,4,4> cheb(a,b,fcn); run_test(cheb);}
  if constexpr (D==3) { ChebFunND<double,3,M,F,4,4,4> cheb(a,b,fcn); run_test(cheb);}
  if constexpr (D==4) { ChebFunND<double,4,M,F,4,4,4,4> cheb(a,b,fcn); run_test(cheb);}
}

template <size_t D, size_t M>
void test_dim_out() {
  xt::xtensor_fixed<double, xt::xshape<D>> a,b;
  for(size_t i=0;i<D;++i){a(i)=-1.0; b(i)=1.0;}
  auto f=[](auto &&xs){
    double s=xt::sum(xs)();
    xt::xtensor_fixed<double, xt::xshape<M>> out;
    for(size_t k=0;k<M;++k) out(k)=(k+1.0)*std::exp(s);
    return out;
  };
  test_interp<D,M>(a,b,f);
}

int main(){
  test_dim_out<1,1>();
  test_dim_out<2,1>();
  test_dim_out<3,1>();
  test_dim_out<2,3>();
  test_dim_out<3,2>();
  test_dim_out<4,4>();
  return 0;
}
