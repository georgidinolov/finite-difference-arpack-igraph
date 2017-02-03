#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <gsl/gsl_sf_exp.h>
#include "1DHeatEquationFiniteDifferenceSolver.hpp"

using namespace std;

int main ()
{
  ContinuousProblemData *cont_data;
  cont_data = new ContinuousProblemData();
  cout << "cont_data=\n" << *cont_data << "\n" << endl;

  QuantizedProblemData *disc_data;
  disc_data = new QuantizedProblemData();
  cout << "disc_data=\n" << *disc_data << "\n" << endl;

  OneDHeatEquationFiniteDifferenceSolver *finite_difference_solver;

  int order = 64;
  double sigma_x = 1.1;
  double a = -1.030155;
  double b = 1.2388;
  double x = 0.097;
  double t = 1;
  finite_difference_solver = 
    new OneDHeatEquationFiniteDifferenceSolver(order,
    					       sigma_x,
    					       a,b,
    					       x,t);


  // double solution = finite_difference_solver->solve();
  // // std::cout << "solution = " << solution << std::endl;
  return 0;
}
