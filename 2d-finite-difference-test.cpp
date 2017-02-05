#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>
#include <gsl/gsl_sf_exp.h>
#include "2DHeatEquationFiniteDifferenceSolver.hpp"
#include "../brownian-motion/2DBrownianMotionPath.hpp"

using namespace std;

int main ()
{
  TwoDHeatEquationFiniteDifferenceSolver *finite_difference_solver;

  unsigned seed = 3;
  int bm_order = 10000;
  int order = 64;
  double rho = 0.0;
  double sigma_x = 1.0;
  double sigma_y = 1.0;
  double t = 1;

  BrownianMotion BM = BrownianMotion(seed,
				     bm_order,
				     rho,
				     sigma_x,
				     sigma_y,
				     0,
				     0,
				     t);
  double a = BM.get_a();
  double b = BM.get_b();
  double c = BM.get_c();
  double d = BM.get_d();
  double x = BM.get_x_T();
  double y = BM.get_y_T();

  printf("(%f,%f,%f) x (%f,%f,%f)\n",a,x,b,c,y,d);

  finite_difference_solver = 
    new TwoDHeatEquationFiniteDifferenceSolver(order,
					       rho,
					       sigma_x,
					       sigma_y,
    					       a,b,
					       c,d,
    					       x,y,
					       t);

  // // double solution = finite_difference_solver->solve();
  // // std::cout << "solution = " << solution << std::endl;


  auto t1 = std::chrono::high_resolution_clock::now();
  double likelihood = finite_difference_solver->solve();
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "duration = "
  	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
  	    << " milliseconds\n";

  printf("likelihood = %.16e, N = %d\n", likelihood, order);
  delete finite_difference_solver;
  return 0;
}
