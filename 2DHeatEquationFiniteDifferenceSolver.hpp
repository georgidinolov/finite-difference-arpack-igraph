// This is the header file for 2DHeatEquationFiniteDifference.cpp It
// contains the interface for the finite difference method for solving
// the heat equation on a bounded domain with absorbing boundary
//
// dq/dt = 1/2 \sigma^2_x d^2 q / dx^2 + \rho \sigma_x sigma_y d^2 q /
// dxdy + 1/2 \sigma^2_y d^2 q / dy^2
//
// q(a,y,t) = q(b,y,t) = 0
// q(x,c,t) = q(x,d,t) = 0
// q(x,y,0 = \delta_{x_0=0, y_0=0}
//
// Note that we are fixing the IC at (0,0).  This class can also take
// derivatives of the heat equation solution witnh respect to the
// boundary values.
//
//    f(a,b,c,d | x_T,y_T,T) = d^4 q / da db dc dd
//

extern "C" {
#include "igraph.h"
}

#include <armadillo>
#include <iostream>
#include <vector>
#include "PDEDataTypes.hpp"

template<class ARFLOAT>
class ARluSymStdEig;

class TwoDHeatEquationFiniteDifferenceSolver
{
public:
  // Instantiates a standard diffusion problem on the unit disk with a
  // Dirac pulse IC at (1/2,1/2) and absorbing boundaries. There is no
  // correlation and the diffusion coeficients are of magnitude 1:
  // 
  // dq/dt = 1/2 d^2 q / dx^2 + 1/2 d^2 q / dy^2
  //
  // q(0,y,t) = q(1,y,t) = 0
  // q(x,0,t) = q(x,1,t) = 0
  // q(x,y,0) = \delta_{x_0 = 1/2, y_0 = 1/2}
  //
  TwoDHeatEquationFiniteDifferenceSolver();

  // Instantiates a standard diffusion problem on a rectangle with Dirac
  // pulse IC and absorbing boundaries. Parameters for the problem are
  // specified by the user for the model:
  // 
  // dq/dt = 1/2 \sigma^2_x d^2 q / dx^2 + \rho \sigma_x sigma_y d^2 q /
  // dxdy + 1/2 \sigma^2_y d^2 q / dy^2
  //
  // q(a,y,t) = q(b,y,t) = 0
  // q(x,c,t) = q(x,d,t) = 0
  // q(x,y,0 = \delta_{x_0, y_0}
  // 
  // Input: 
  //     x_initial_condition = x_0,
  //     y_initial_condition = y_0,
  //     rho = rho,
  //     sigma_x = sigma_x,
  //     sigma_y = sigma_y,
  //     a = a, b = b, c = c, d = d
  //
  // Output: a solver for the diffusion PDE, as well as the ability to calculate 
  //    f(a,b,c,d | x_T,y_T,T) = d^4 q / da db dc dd
  //
  TwoDHeatEquationFiniteDifferenceSolver(int order,
  					 double rho,
  					 double sigma_x,
  					 double sigma_y,
  					 double a, 
  					 double b,
  					 double c, 
  					 double d,
  					 double x_t,
  					 double y_t,
  					 double t);
  ~TwoDHeatEquationFiniteDifferenceSolver();
  
  // const ContinuousProblemData& get_quantized_continuous_data() const;
  // const DiscreteProblemData& get_quantized_discrete_data() const;

  // // Returns std::vector<int> of quantized boundary values, given
  // // interior (unscaled) point (x,y)
  // // Input:
  // //     double x: unscaled x coordinate
  // //     double y: unscaled y coordinate
  // // Output: vector of index integers of boundary values
  // //     std::vector<int> (i_a, i_b, i_c, i_d)
  // std::vector<int> get_quantized_boundary_values(double x, double y) const;
  
  // // Returns a pair of indeces for interior point
  // std::vector<int> get_quantized_interior_point(double x, double y) const;

  // double get_sigma_x() const;
  // double get_sigma_y() const;

  void set_switch_x_y();
  void set_sigma_x(double sigma_x);
  void set_sigma_y(double sigma_y);

  // the solution to the heat equation at (x_T,y_T) for time T
  double solve();

  // the solution to the heat equation at (x_T,y_T) for time T
  double solve(double x_T, double y_T, double T);
  
  double likelihood();

  // // // the solution to the heat equation at the collections of pairs
  // // // (x_T,y_T) for time T
  // // std::vector<double> solve(double T, 
  // // 			    std::vector<double> x_T, 
  // // 			    std::vector<double> y_T);

  // // // the solution to the heat equation at the collections of pairs
  // // // (x_T,y_T) for all times T
  // // std::vector<std::vector<double>> solve(std::vector<double> T, 
  // // 					 std::vector<double> x_T, 
  // // 					 std::vector<double> y_T);
  
  

private:
  // BC, interior point for the problem
  ContinuousProblemData original_data_;
  // BC, interior point for the problem, scaled according to STEP 1
  // and STEP 2
  ContinuousProblemData scaled_data_;
  ContinuousProblemData quantized_continuous_data_;
  DiscreteProblemData quantized_discrete_data_;
  // iL, iR, jL, jU
  BoundaryIndeces boundary_indeces_;
  // 
  std::vector<arma::mat> eigenvectors_;
  std::vector<arma::vec> eigenvalues_;
  double alpha_;
  QuantizedProblemData quantized_shifted_data_;

  // 1/order = Delta x = Delta y in the discrete scheme
  int order_;
  // correlation coefficient 
  double rho_;
  // diffusion coefficient in the x-dirction
  double sigma_x_; 
  // diffusion coefficient in the y-dirction
  double sigma_y_;
  double correction_factor_;
  // whether to consider (x,y) or (y,x) in the problem
  bool switch_x_y_;
  // The first and second of two system matrices for this problem,
  // denoted as S1 in the documentation.
  arma::Mat<double> system_matrix_one_;
  arma::Mat<double> system_matrix_two_;
  arma::Mat<arma::uword> index_matrix_;
  arma::vec system_matrix_;
  std::vector<ARluSymStdEig<double>*> eigenprobs_;

  // Switches the role of x- and y-coordinates if necessary, then
  // scales the boundary data by sigma_x_ and sigma_y_, then scales
  // the boundary data by the boundary ranges. This corresponds to
  // STEP 1 and STEP 2 in the write-up.
  void scale_data();

  // Places scaled boundaries and (x,y) on the discrete index for the
  // problem. This corresponds to STEP 3 in the write-up. Sets the private object 
  // quantized_data_;
  void quantize_data();

  // pre-calculates matrices S1, S2, and I. Those are independent of
  // which extended region we use so they can be built that way.
  void pre_calc_S_matrices();

  // pre-calculates system matrix; gets called by build_final_matrix.
  const SystemMatrices pre_calc_system_matrix(int i_L, int i_R, int j_L, int j_U) const;

  // solves the eigenvalue problem and stores it in the private member
  ARluSymStdEig<double> * solve_eigenproblem(unsigned i_L, 
					     unsigned i_R, 
					     unsigned j_L, 
					     unsigned j_U) const;

  // builds final matrix
  double solve_discretized_PDE(unsigned i_L, 
			       unsigned i_R, 
			       unsigned j_L, 
			       unsigned j_U,
			       ARluSymStdEig<double> * eigenproblem_ptr) const;

  int number_eigenvalues(double t_2, double Delta_cut) const;
  bool check_data(int n, int* pcol, int* irow, char uplo) const;
};

class Eigenproblem
{
public:
  Eigenproblem(igraph_vector_t * eigenvalues_ptr,
	       igraph_matrix_t * eigenvectors_ptr);
  ~Eigenproblem();
private:
  igraph_vector_t* eigenvalues_ptr_;
  igraph_matrix_t* eigenvectors_ptr_;
};
