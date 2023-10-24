#include "nlsolver.h"
#include "stopwatch.h"

#include "cppnumericaloptimizers/eigen.h"
#include "cppnumericaloptimizers/include/cppoptlib/function.h"
#include "cppnumericaloptimizers/include/cppoptlib/solver/bfgs.h"
using FunctionXd = cppoptlib::function::Function<double>;

using nlsolver::BFGS;
// Rosenbrock for nlsolver
class Rosenbrock {
public:
  double operator()(std::vector<double> &x) {
    const double t1 = 1 - x[0];
    const double t2 = (x[1] - x[0] * x[0]);
    return t1 * t1 + 100 * t2 * t2;
  }
};
class Rosenbrock2 : public FunctionXd {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  double operator()(const Eigen::VectorXd &x) const {
    const double t1 = (1 - x[0]);
    const double t2 = (x[1] - x[0] * x[0]);
    return   t1 * t1 + 100 * t2 * t2;
  }
};

void print_vector(std::vector<double> &x) {
  for (auto &val : x) {
    std::cout << val << ",";
  }
  std::cout << "\n";
}

int main() {
  using clocktype = decltype(Stopwatch()());
  // define problem functor - in our case a variant of the rosenbrock function
  Rosenbrock prob;
  std::vector<double> bfgs_init = {2, 2};
  std::cout << "BFGS" << std::endl;
  clocktype timing1;
  [&](){
    Stopwatch sw;
    auto bfgs_solver = BFGS<Rosenbrock, double>(prob);
    auto bfgs_res = bfgs_solver.minimize(bfgs_init);
    //bfgs_res.print();
    timing1 = sw();
  }();
  print_vector(bfgs_init);
  //
  cppoptlib::solver::Bfgs<Rosenbrock2> solver;
  Rosenbrock2 prob2;
  Eigen::VectorXd x(2);
  x << 2, 2;
  // and finally, minimize
  clocktype timing2;
  [&](){
    Stopwatch sw;
    auto [solution, solver_state] = solver.Minimize(prob2, x);
    timing2 = sw();
    //std::cout << "argmin " << solution.x.transpose() << std::endl;
  }();
  std::cout <<"nlsolver "<< timing2/timing1 <<" times faster.\n";
  return 0;
}