// This is the header file for 1DHeatEquationFiniteDifference.cpp It
// contains the interface for the finite difference method for solving
// the heat equation on a bounded domain with absorbing boundary
//
// dq/dt = 1/2 \sigma^2_x d^2 q / dx^2
//
// q(a,t) = q(b,t) = 0
// q(x,0) = \delta_{x_0=0}
//
// Note that we are fixing the IC at (0,0). 
//
// This class can also take derivatives of the heat equation solution
// witnh respect to the boundary values.
//
//    f(a,b,c,d | x_T,y_T,T) = d^4 q / da db dc dd
//
#include <armadillo>
#include <iostream>
#include <vector>

class ContinuousProblemData
{
public:
  ContinuousProblemData();
  ContinuousProblemData(double x_T,
			double x_0,
			double t,
			double a,
			double b);

  double get_x_T() const;
  void set_x_T(double x_T);

  double get_x_0() const;

  double get_t() const;
  void set_t(double t);

  double get_a() const;
  double get_b() const;
  friend std::ostream& operator <<(std::ostream& output_stream, 
				   const ContinuousProblemData& c_prod_data);

private:
  double x_T_;
  double x_0_;
  double t_;
  double a_;
  double b_;
};

class QuantizedProblemData
{
public:
  QuantizedProblemData();
  QuantizedProblemData(int i_T, 
		       int i_0,
		       int i_L, 
		       int i_R);

  int get_i_T() const;
  int get_i_0() const;
  int get_i_L() const;
  int get_i_R() const;
  friend std::ostream& operator <<(std::ostream& output_stream, 
				   const QuantizedProblemData& q_prod_data);
private:
  int i_T_;
  int i_0_;
  int i_L_;
  int i_R_;
};

class OneDHeatEquationFiniteDifferenceSolver
{
public:
  // Instantiates a standard diffusion problem on the unit interval
  // [-1/2,1/2] with a Dirac pulse IC at 0 and absorbing
  // boundaries. There is no correlation and the diffusion coeficients
  // are of magnitude 1:
  //
  // dq/dt = 1/2 d^2 q / dx^2
  //
  // q(-1/2,t) = q(1/2,t) = 0
  // q(x,0) = \delta_{x_0 = 0}
  //
  OneDHeatEquationFiniteDifferenceSolver();

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
  OneDHeatEquationFiniteDifferenceSolver(int order,
					 double sigma_x,
					 double a, 
					 double b,
					 double x_t,
					 double t);
  ~OneDHeatEquationFiniteDifferenceSolver();

  // the solution to the heat equation at (x_T,y_T) for time T
  double solve();

//   // the solution to the heat equation at (x_T,y_T) for time T
//   double solve(double x_T, double y_T, double T);
  
  double likelihood();

//   // // the solution to the heat equation at the collections of pairs
//   // // (x_T,y_T) for time T
//   // std::vector<double> solve(double T, 
//   // 			    std::vector<double> x_T, 
//   // 			    std::vector<double> y_T);

//   // // the solution to the heat equation at the collections of pairs
//   // // (x_T,y_T) for all times T
//   // std::vector<std::vector<double>> solve(std::vector<double> T, 
//   // 					 std::vector<double> x_T, 
//   // 					 std::vector<double> y_T);
  
  

private:
  // BC, interior point for the problem
  ContinuousProblemData original_data_;
  // BC, interior point for the problem, scaled according to STEP 1
  // and STEP 2
  ContinuousProblemData scaled_data_;
  // BC, interior point quantized according to STEP 3
  QuantizedProblemData quantized_data_;
  // BC, interior point shifted according to STEP 4
  QuantizedProblemData quantized_shifted_data_;

  // 1/order = Delta x = Delta y in the discrete scheme
  int order_;
  // diffusion coefficient in the x-dirction
  double sigma_x_; 
  // The first and second of two system matrices for this problem,
  // denoted as S1 in the documentation.
  arma::vec* system_matrix_one_;
  arma::Mat<arma::uword>* index_matrix_;
  arma::vec* system_matrix_;

  // Scales the boundary data by sigma_x_, then scales
  // the boundary data by the boundary range. This corresponds to
  // STEP 1 and STEP 2 in the write-up.
  void scale_data();

  // Places scaled boundaries and (x) on the discrete index for the
  // problem. This corresponds to STEP 3 in the write-up. Sets the private object 
  // quantized_data_;
  void quantize_data();

  // Shifts quantized data so that the lower end of the
  // extended computational region is with index >= 1
  void shift_quantized_data();

  // pre-calculates matrices S1, S2, and I. Those are independent of
  // which extended region we use so they can be built that way.
  void pre_calc_S_matrices();

  // pre-calculates system matrix; gets called by build_final_matrix.
  arma::umat pre_calc_system_matrix(int i_L, int i_R);

  // builds final matrix
  double solve_discretized_PDE(unsigned i_L, 
 			       unsigned i_R);
  
  int number_eigenvalues(double t_2, double Delta_cut) const;
  bool check_data(int n, int* pcol, int* irow, char uplo) const;
};

