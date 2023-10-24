#include "nlsolver.h"
#include "stopwatch.h"

#include "cppnumericaloptimizers/eigen.h"
#include "cppnumericaloptimizers/include/cppoptlib/function.h"
#include "cppnumericaloptimizers/include/cppoptlib/solver/bfgs.h"
using FunctionXd = cppoptlib::function::Function<double>;

using nlsolver::BFGS;

class Arma : public FunctionXd {
  std::vector<double> y;
  size_t p,q,n;
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Arma(const std::vector<double> &y,
       const size_t p, const size_t q) : y(y), p(p), q(q), n(y.size()){}
  double operator()(const Eigen::VectorXd &x) const {
    // we do this only because otherwise we would have to modify cppnumericalsolver
    // to make this method nonconst
    // sadly constexpr array::...::fill is C++20 only
    std::array<double, 100> resid;
    resid[0] = 0;
    resid[1] = 0;
    int ma_offset;
    double ssq = 0.0, tmp = 0.0;
    for (size_t l = p; l < n; l++) {
      ma_offset = std::min(l-p, q);
      tmp = y[l];
      for (size_t j = 0; j < p; j++) {
        tmp -= x[j] * y[l - j - 1];
      }
      // to offset that this is all in one vector, we need to
      // start at p and go to p + q
      for (size_t j = 0; j < ma_offset; j++) {
        tmp -= x[p + j] * resid[l - j - 1];
      }
      resid[l] = tmp;
      ssq += tmp * tmp;
    }
    return 0.5 * std::log(ssq / n);
  }
};

void print_vector(std::vector<double> &x) {
  for (auto &val : x) {
    std::cout << val << ",";
  }
  std::cout << "\n";
}

int main() {
  std::cout << "Running ARMA(2,1) without intercept in R as: \n"<<
      " 'arima(lynx[1:100], order =c(2,0,1),include.mean = F)' \n" <<
      " yields coefficients: 0.9319,  -0.2051,  0.4152 \n" <<
      " we should hopefully reproduce those.\n";
  using clocktype = decltype(Stopwatch()());
  // define problem functor - nlsolver actually supports lambdas
  const std::vector<double> y = {
      269, 321, 585, 871, 1475, 2821, 3928, 5943, 4950, 2577, 523,
      98, 184, 279, 409, 2285, 2685, 3409, 1824, 409, 151, 45, 68,
      213, 546, 1033, 2129, 2536, 957, 361, 377, 225, 360, 731, 1638,
      2725, 2871, 2119, 684, 299, 236, 245, 552, 1623, 3311, 6721,
      4254, 687, 255, 473, 358, 784, 1594, 1676, 2251, 1426, 756, 299,
      201, 229, 469, 736, 2042, 2811, 4431, 2511, 389, 73, 39, 49,
      59, 188, 377, 1292, 4031, 3495, 587, 105, 153, 387, 758, 1307,
      3465, 6991, 6313, 3794, 1836, 345, 382, 808, 1388, 2713, 3800,
      3091, 2985, 3790, 674, 81, 80, 108
  };
  size_t p = 2, q = 2, n = 100;
  //std::vector<double> resid(n, 0.0);
  auto arma_lam = [&](std::vector<double> &x) {
    std::array<double, 100> resid;
    resid[0] = 0;
    resid[1] = 0;
    // simple ARMA(2,2) model; we fix the p and q here
    // just to simplify, even though this would not
    // be done in the real world
    int ma_offset;
    double ssq = 0.0, tmp = 0.0;
    for (size_t l = p; l < n; l++) {
      ma_offset = std::min(l - p, q);
      tmp = y[l];
      for (size_t j = 0; j < p; j++) {
        tmp -= x[j] * y[l - j - 1];
      }
      // to offset that this is all in one vector, we need to
      // start at p and go to p + q
      for (size_t j = 0; j < ma_offset; j++) {
        tmp -= x[p + j] * resid[l - j - 1];
      }
      resid[l] = tmp;
      ssq += tmp * tmp;
    }
    return 0.5 * std::log(ssq / n);
  };

  std::vector<double> bfgs_init(p+q,0.0);
  std::cout << "nlsolver BFGS" << std::endl;
  clocktype timing1;
  [&](){
    Stopwatch sw;
    auto bfgs_solver = BFGS<decltype(arma_lam), double>(arma_lam);
    auto bfgs_res = bfgs_solver.minimize(bfgs_init);
    timing1 = sw();
    print_vector(bfgs_init);
    bfgs_res.print();
  }();
  //
  cppoptlib::solver::Bfgs<Arma> solver;
  // initialize outside of timed lambda
  Arma prob2(y,p,q);
  Eigen::VectorXd x = Eigen::VectorXd::Zero(p+q);
  // and finally, minimize
  clocktype timing2;
  [&](){
    Stopwatch sw;
    auto [solution, solver_state] = solver.Minimize(prob2, x);
    timing2 = sw();
    std::cout << solution.x.transpose() << std::endl;
    std::cout << "f in argmin " << solution.value << std::endl;
    std::cout << "iterations " << solver_state.num_iterations << std::endl;
    std::cout << "solver status " << solver_state.status << std::endl;
  }();
  std::cout <<"nlsolver "<< timing2/timing1 <<" times faster.\n";

  return 0;
}