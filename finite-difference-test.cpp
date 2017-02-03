#include <armadillo>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <gsl/gsl_sf_exp.h>
#include "2DHeatEquationFiniteDifferenceSolver.hpp"

using namespace arma;
using namespace std;

int main ()
{
  TwoDHeatEquationFiniteDifferenceSolver *finite_difference_solver;

  int order = 256;
  double rho = 0.4;
  double sigma_x = 1.1;
  double sigma_y = 0.9;
  double a = -0.030155;
  double b = 1.2388;
  double x = 1.1097;
  double c = -0.053901;
  double d = 0.50645;
  double y = 0.34923;
  double t = 1;
  finite_difference_solver = 
    new TwoDHeatEquationFiniteDifferenceSolver(order,
  					       rho,
  					       sigma_x,
  					       sigma_y,
  					       a,b,c,d,
  					       x,y,t);

  double solution = finite_difference_solver->solve();
  std::cout << "solution = " << solution << std::endl;
  // finite_difference_solver->set_sigma_x(0.5);
  // derivative = finite_difference_solver->likelihood();
  // std::cout << "derivative = " << derivative << std::endl;


  // finite_diff_test.open("/home/gdinolov/Research/PDE-solvers/src/finite-difference/finite-diff-test.csv");
  // // Writing header.
  // finite_diff_test << "i-index" << ",";
  // finite_diff_test << "j-index";
  // finite_diff_test << "\n";
  
  // // Writing the i and j indeces.
  // for (int i=quantized_data.get_i_L(); i<=quantized_data.get_i_R(); ++i) {
  //   for (int j=quantized_data.get_j_L(); j<=quantized_data.get_j_U(); ++j) {
  //     finite_diff_test << i << "," << j;
  //     finite_diff_test << "\n";
  //   }
  // }

  // finite_diff_test.close();

  delete finite_difference_solver;
  return 0;
}
