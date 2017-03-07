#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>
#include <gsl/gsl_sf_exp.h>
#include "2DHeatEquationFiniteDifferenceSolver.hpp"
#include "../brownian-motion/2DBrownianMotionPath.hpp"

using namespace std;

int main(int argc, char *argv[])
{
  if (argc < 12) {
    printf("You must provide input\n");
    printf("The input is: \n int order of numerical accuracy (try 32, 64, or 128 for now); \n sigma_x, \n sigma_y, \n rho,\n t,\n a,\n x_T,\n b,\n c,\n y_T,\n d\n"); 
    exit(0);
  }

  TwoDHeatEquationFiniteDifferenceSolver *finite_difference_solver;

  int order = std::stoi(argv[1]);
  double sigma_x = std::stod(argv[2]);
  double sigma_y = std::stod(argv[3]);
  double rho = std::stod(argv[4]);
  double t = std::stod(argv[5]);
  double a = std::stod(argv[6]);
  double x = std::stod(argv[7]);
  double b = std::stod(argv[8]);
    
  double c = std::stod(argv[9]);
  double y = std::stod(argv[10]);
  double d = std::stod(argv[11]);

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
  double solution = finite_difference_solver->likelihood();
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "duration = "
  	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
  	    << " milliseconds\n";
  printf("solution = %.16e, N = %d\n", solution, order);
  finite_difference_solver->save_data_point();
  
  delete finite_difference_solver;
  return 0;
}
