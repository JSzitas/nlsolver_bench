#include "eigen.h"
#include "include/cppoptlib/function.h"
#include "include/cppoptlib/solver/bfgs.h"

using FunctionXd = cppoptlib::function::Function<double>;

    class Rosenbrock : public FunctionXd {
      public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        double operator()(const Eigen::VectorXd &x) const {
            const double t1 = (1 - x[0]);
            const double t2 = (x[1] - x[0] * x[0]);
            return   t1 * t1 + 100 * t2 * t2;
        }
    };
    int main(int argc, char const *argv[]) {
        using SolverBFGS = cppoptlib::solver::Bfgs<Rosenbrock>;
		using SolverGD = cppoptlib::solver::GradientDescent<Rosenbrock>;
		using SolverCGD = cppoptlib::solver::ConjugatedGradientDescent<Rosenbrock>;


        Rosenbrock f;
        Eigen::VectorXd x(2);
        x << 2, 2;

        // Evaluate
        //auto state = f.Eval(x);
        //std::cout << f(x) << " = " << state.value;// << std::endl;
        //std::cout << " coef: " <<state.x;// << std::endl;
        //std::cout << " gradient: " << state.gradient << std::endl;
        //if (state.hessian) {
        //  std::cout << *(state.hessian) << std::endl;
        //}

        //std::cout << cppoptlib::utils::IsGradientCorrect(f, x) << std::endl;
        //std::cout << cppoptlib::utils::IsHessianCorrect(f, x) << std::endl;
		[&](){
			SolverBFGS solver;
        	auto [solution, solver_state] = solver.Minimize(f, x);
        	std::cout << "argmin " << solution.x.transpose() << std::endl;
        	std::cout << "f in argmin " << solution.value << std::endl;
        	std::cout << "iterations " << solver_state.num_iterations << std::endl;
        	std::cout << "solver status " << solver_state.status << std::endl;
		}();
		[&](){
			SolverGD solver;
        	auto [solution, solver_state] = solver.Minimize(f, x);
        	std::cout << "argmin " << solution.x.transpose() << std::endl;
        	std::cout << "f in argmin " << solution.value << std::endl;
        	std::cout << "iterations " << solver_state.num_iterations << std::endl;
        	std::cout << "solver status " << solver_state.status << std::endl;
		};
		[&](){
			SolverCGD solver;
        	auto [solution, solver_state] = solver.Minimize(f, x);
        	std::cout << "argmin " << solution.x.transpose() << std::endl;
        	std::cout << "f in argmin " << solution.value << std::endl;
        	std::cout << "iterations " << solver_state.num_iterations << std::endl;
        	std::cout << "solver status " << solver_state.status << std::endl;
		};



        return 0;
    }
