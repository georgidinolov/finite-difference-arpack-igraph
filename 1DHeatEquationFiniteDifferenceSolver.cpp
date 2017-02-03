// This is the implementation for the finite difference method for
// solving the heat equation on a bounded domain with absorbing
// boundary
//
// dq/dt = 1/2 \sigma^2_x d^2 q / dx^2 + \rho \sigma_x sigma_y d^2 q /
// dxdy + 1/2 \sigma^2_y d^2 q / dy^2
//
// q(a,y,t) = q(b,y,t) = 0
// q(x,c,t) = q(x,d,t) = 0
// q(x,y,0 = \delta_{x_0 = 0, y_0 = 0}
// 
// This class can also take derivatives of the heat equation solution
// with respect to the boundary values. 
//
//    f(a,b,c,d | x_T,y_T,T) = d^4 q / da db dc dd
//
//

#include <algorithm>
#include <armadillo>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include "1DHeatEquationFiniteDifferenceSolver.hpp"
#include "arpackpp/include/arlsmat.h"
#include "arpackpp/include/arlssym.h"
#include "arpackpp/examples/matrices/sym/lsmatrxb.h"

using namespace arma;

namespace{ 
  inline double square(double x) {
    return x * x;
  }

  inline double cubed(double x) {
    return x * x * x;
  }
  
  inline double pi() {
    return std::atan(1)*4;
  }
}

ContinuousProblemData::ContinuousProblemData()
  : x_T_(0.5),
    x_0_(0.5),
    t_(1.0),
    a_(0.0),
    b_(1.0)
{}

ContinuousProblemData::ContinuousProblemData(double x_T,
					     double x_0,
					     double t,
					     double a,
					     double b)
  : x_T_(x_T),
    x_0_(x_0),
    t_(t),
    a_(a),
    b_(b)
{}

double ContinuousProblemData::get_x_T() const 
{
  return x_T_;
}
void ContinuousProblemData::set_x_T(double x_T) 
{
  x_T_ = x_T;
}
double ContinuousProblemData::get_x_0() const 
{
  return x_0_;
}
double ContinuousProblemData::get_t() const 
{
  return t_;
}
void ContinuousProblemData::set_t(double t)
{
  t_ = t;
}
double ContinuousProblemData::get_a() const
{
  return a_;
}
double ContinuousProblemData::get_b() const
{
  return b_;
}

std::ostream& operator<<(std::ostream& output_stream, 
			  const ContinuousProblemData& c_prod_data) 
{
  output_stream << "t=" << c_prod_data.t_ << "\n";
  output_stream << "x_T=" << c_prod_data.x_T_ << "\n";
  output_stream << "x_0=" << c_prod_data.x_0_ << "\n";
  output_stream << "(a,b)=" << "(" << 
    c_prod_data.a_ << "," << c_prod_data.b_ << ")";
  return output_stream;
}


QuantizedProblemData::QuantizedProblemData()
  : i_T_(0),
    i_0_(0),
    i_L_(0),
    i_R_(1)
{}

QuantizedProblemData::QuantizedProblemData(int i_T, 
					   int i_0,
					   int i_L, 
					   int i_R)
  : i_T_(i_T),
    i_0_(i_0),
    i_L_(i_L),
    i_R_(i_R)
{}

int QuantizedProblemData::get_i_T() const {
  return i_T_;
}
int QuantizedProblemData::get_i_0() const {
  return i_0_;
}
int QuantizedProblemData::get_i_L() const {
  return i_L_;
}
int QuantizedProblemData::get_i_R() const {
  return i_R_;
}

std::ostream& operator<<(std::ostream& output_stream, 
			  const QuantizedProblemData& q_prod_data) 
{
  output_stream << "i_T=" << q_prod_data.i_T_ << "\n";
  output_stream << "i_0=" << q_prod_data.i_0_ << "\n";
  output_stream << "(i_L,i_R)=" << "(" << 
    q_prod_data.i_L_ << "," <<
    q_prod_data.i_R_ << ")";
  return output_stream;
}

OneDHeatEquationFiniteDifferenceSolver::OneDHeatEquationFiniteDifferenceSolver()
  : original_data_(ContinuousProblemData(0, 0, 1, -0.5, 0.5)),
    order_(64),
    sigma_x_(1.0)
{
  scale_data();
  quantize_data();
  shift_quantized_data();
  pre_calc_S_matrices();
}

OneDHeatEquationFiniteDifferenceSolver::
OneDHeatEquationFiniteDifferenceSolver(int order,
				       double sigma_x,
				       double a, 
				       double b,
				       double x_t, 
				       double t)
  : original_data_(x_t,0,t,a,b),
    order_(order),
    sigma_x_(sigma_x),
    system_matrix_one_(new arma::vec(1)),
    index_matrix_(new arma::Mat<arma::uword>(1,1)),
    system_matrix_(new arma::vec(1))
{
  // TODO(gdinolov) : I need to throw an exception if (x) is outside
  // the boundary.

  // Steps 1 and 2 in the write-up.
  scale_data();
  // Step 3 in the write-up
  quantize_data();
  // Step 6 in the write-up
  shift_quantized_data();
  pre_calc_S_matrices();
}

OneDHeatEquationFiniteDifferenceSolver::
~OneDHeatEquationFiniteDifferenceSolver() {
  delete system_matrix_one_;
  delete index_matrix_;
  delete system_matrix_;
}

double OneDHeatEquationFiniteDifferenceSolver::solve() 
{
  double solution = 1.0 / (original_data_.get_b() - original_data_.get_a()) *
    solve_discretized_PDE(quantized_shifted_data_.get_i_L(),
			  quantized_shifted_data_.get_i_R());

  return solution;
}

// double TwoDHeatEquationFiniteDifferenceSolver::solve(double x_T, 
// 						     double y_T, 
// 						     double T) 
// {
//   // TODO(georgid): We need to check if the new point is in between
//   // the boundaries
//   original_data_.set_x_T(x_T);
//   original_data_.set_y_T(y_T);
//   original_data_.set_t(T);
//   scale_data();
//   quantize_data();
//   shift_quantized_data();
//   pre_calc_S_matrices();
  
//   double solution = solve();
//   return solution;
// }

double OneDHeatEquationFiniteDifferenceSolver::likelihood() 
{
  double sol_1 = solve_discretized_PDE(quantized_shifted_data_.get_i_L()-1,
				       quantized_shifted_data_.get_i_R()+1);

  double sol_2 = solve_discretized_PDE(quantized_shifted_data_.get_i_L(),
				       quantized_shifted_data_.get_i_R()+1);

  double sol_3 = solve_discretized_PDE(quantized_shifted_data_.get_i_L()-1,
				       quantized_shifted_data_.get_i_R());

  double sol_4 = solve_discretized_PDE(quantized_shifted_data_.get_i_L(),
				       quantized_shifted_data_.get_i_R());

  double h2_inv = square(1.0*order_);

  double derivative = -1 * h2_inv *
    1.0 / pow(original_data_.get_b() - original_data_.get_a(),3) * 
    (-1.0*sol_1 + sol_2 + sol_3 - sol_4);

  return derivative;
}

void OneDHeatEquationFiniteDifferenceSolver::scale_data() {

  double scaled_T = original_data_.get_t() * square(sigma_x_) /
    square(original_data_.get_b() - original_data_.get_a());

  double scaled_x_T = original_data_.get_x_T() / 
    (original_data_.get_b() - original_data_.get_a());
  double scaled_x_0 = original_data_.get_x_0() / 
    (original_data_.get_b() - original_data_.get_a());
  double scaled_a = original_data_.get_a() / 
    (original_data_.get_b() - original_data_.get_a());
  double scaled_b = original_data_.get_b() / 
    (original_data_.get_b() - original_data_.get_a());

  scaled_data_ = ContinuousProblemData(scaled_x_T,
				       scaled_x_0,
				       scaled_T,
				       scaled_a,
				       scaled_b);
}

void OneDHeatEquationFiniteDifferenceSolver::quantize_data()
{
  int i_T = std::round( scaled_data_.get_x_T()
			/ (1.0/order_) );

  // index for a_
  int a_shifted_index = std::round( 1.0*scaled_data_.get_a()/(1.0/order_) + 0.5);
  std::vector<int> i_a_values = {0, i_T, a_shifted_index};
  std::vector<int>::iterator min_i_a = std::min_element(std::begin(i_a_values),
							std::end(i_a_values));
  int i_L = *min_i_a;

  // index for b_
  int b_shifted_index = std::round( 1.0*scaled_data_.get_b()/(1.0/order_) - 0.5);
  std::vector<int> i_b_values = {0, i_T, b_shifted_index};
  std::vector<int>::iterator max_i_b = std::max_element(std::begin(i_b_values),
							std::end(i_b_values));
  int i_R = *max_i_b;

  quantized_data_ = QuantizedProblemData(i_T,
					 0,
					 i_L,
					 i_R);
}

void OneDHeatEquationFiniteDifferenceSolver::shift_quantized_data() {
  int i_not = -1*(quantized_data_.get_i_L()-1);

  int i_T = quantized_data_.get_i_T() + i_not;

  int i_L = quantized_data_.get_i_L() + i_not;
  int i_R = quantized_data_.get_i_R() + i_not;

  quantized_shifted_data_ = QuantizedProblemData(i_T,
						 i_not,
						 i_L,
						 i_R);
}

void OneDHeatEquationFiniteDifferenceSolver::pre_calc_S_matrices()
{
  unsigned N = order_ + 2;
  index_matrix_->resize(3*N,2);
  system_matrix_one_->resize(3*N);
  
  int k_0 = 0;
  
  for (unsigned i=1; i<=N; ++i) {
    index_matrix_->row(k_0) = arma::Row<arma::uword> {i,i};
    index_matrix_->row(k_0+1) = arma::Row<arma::uword> {i,i-1};
    index_matrix_->row(k_0+2) = arma::Row<arma::uword> {i,i+1};
    
    system_matrix_one_->at(k_0) = -2.0;
    system_matrix_one_->at(k_0+1) = 1.0;
    system_matrix_one_->at(k_0+2) = 1.0;
    
    k_0 = k_0+3;
  }
}

arma::umat OneDHeatEquationFiniteDifferenceSolver::
pre_calc_system_matrix(int i_L, int i_R)
{
  // std::cout << "In pre_calc_system_matrix(int i_L, int i_R)" << std::endl;
  // std::cout << "original_data_:\n" << original_data_ << std::endl;
  // std::cout << "scaled_data_:\n" << scaled_data_ << std::endl;
  // std::cout << "quantized_data_:\n" << quantized_data_ << std::endl;
  // std::cout << "quantized_shifted_data_:\n" << quantized_shifted_data_ 
  // 	    << std::endl;

  double h2 = square(1.0/order_);
  arma::uvec ind = 
    find(index_matrix_->col(0) > i_L && 
  	 index_matrix_->col(0) < i_R && 
  	 index_matrix_->col(1) > i_L && 
  	 index_matrix_->col(1) < i_R);

  arma::uvec all_columns = {0,1};
  arma::umat index_matrix = index_matrix_->elem(ind,all_columns);

  int sys_matrix_size = ind.n_rows;
  system_matrix_->resize(sys_matrix_size);
  for (int i=0; i<sys_matrix_size; ++i) {
    system_matrix_->at(i) = 
      (1.0/(2*h2)) * system_matrix_one_->at(ind(i));
  }

  return index_matrix;
}

double OneDHeatEquationFiniteDifferenceSolver::
solve_discretized_PDE(unsigned i_L, 
		      unsigned i_R)
{
  if (quantized_shifted_data_.get_i_0() == i_L ||
      quantized_shifted_data_.get_i_0() == i_R ||
      quantized_shifted_data_.get_i_T() == i_L ||
      quantized_shifted_data_.get_i_T() == i_R) {
    return 0;
  } else {
    arma::umat index_matrix = pre_calc_system_matrix(i_L, i_R);
    
    arma::Mat<arma::uword> locations(2,index_matrix.n_rows);
    locations.row(0) = index_matrix.col(0).t() - i_L - 1;
    locations.row(1) = index_matrix.col(1).t() - i_L - 1;
    
    int np = (i_R-1)-(i_L+1)+1;
    
    // std::cout << "np=" << np << std::endl;
    // std::cout << "system_matrix_.n_rows=" << system_matrix_->n_rows << std::endl;
    
    // arma::mat dense_system_matrix = zeros(np,np);
    // for (unsigned i=0; i<locations.n_cols; ++i) {
    //   dense_system_matrix( locations(0,i), locations(1,i) ) = 
    //     system_matrix_->at(i);
    // }
    // std::cout << "dense_system_matrix=\n" << dense_system_matrix << std::endl;
    
    // condition locations.row(1) >= locations.row(0) means that we are
    // considering the upper diagonal entries in the system matrix.
    arma::uvec upper_diag_elements = find(locations.row(1) >= locations.row(0));
    int n = (i_R-1)-(i_L+1)+1;
    int nnz = upper_diag_elements.n_rows;
    
    // std::cout << "nnz=" << nnz << std::endl;
    // std::cout << "n=" << n << std::endl;
    int *irow = new int[nnz];
    int* pcol = new int[n+1];
    double* A = new double[nnz];
    
    unsigned current_row = locations(0,0);
    int column_counter = 0;
    int column_size_counter = 0;
    pcol[column_counter] = 0; // the first element in A is the beginning of the first column
    // std::cout << "current_row=" << current_row << std::endl;

    for (unsigned i=0; i<upper_diag_elements.n_rows; ++i) {
    
      // std::cout << "(" << locations(0,upper_diag_elements(i)) 
      // 	      << "," << locations(1,upper_diag_elements(i))
      // 	      << ") = " << system_matrix_->at(upper_diag_elements(i)) << "; ";

      // if current_row is not the row for element i, we have moved to a
      // new row
      if (current_row != locations(0,upper_diag_elements(i))) {
	if (column_size_counter == 0) {
	  // nothing happens at the beginning of the loop.
	}
	else if (column_size_counter == 1) {
	  A[i-1] = system_matrix_->at(upper_diag_elements(i-1));
	  irow[i-1] = locations(1,upper_diag_elements(i-1));
	} else if (column_size_counter == 2) {
	  A[i-2] = system_matrix_->at(upper_diag_elements(i-2));	
	  A[i-1] = system_matrix_->at(upper_diag_elements(i-1));

	  irow[i-2] = locations(1,upper_diag_elements(i-2));	
	  irow[i-1] = locations(1,upper_diag_elements(i-1));
	} else {
	  // // TODO(georgid): Throw an exception!!
	  // std::cout << "In constructing matrix, shouldn't hit this" << std::cout;
	}

	current_row = locations(0,upper_diag_elements(i));
	column_counter++;
	pcol[column_counter] = i;
	// std::cout << "new column in transpose; column #=" << column_counter+1
	//  		<< "; pcol[column_counter]=" << pcol[column_counter];      
	column_size_counter = 1;

      } else {
	column_size_counter++;
      }

      if (i == upper_diag_elements.n_rows-1) {
	A[i] = system_matrix_->at(upper_diag_elements(i));
	irow[i] = locations(1,upper_diag_elements(i));	
      }
      // std::cout << "\n";
    }
    //  std::cout << std::endl;

    // std::cout << "A = {";
    // for (int i=0; i<nnz; ++i) {
    //   if (i != nnz-1) {
    //     std::cout << A[i] << ", ";
    //   } else {
    //     std::cout << A[i] << "}" << std::endl; 
    //   }
    // }

    pcol[column_counter+1]=nnz;
    // std::cout << "pcol = {";
    // for (int i=0; i<n+1; ++i) {
    //     std::cout << pcol[i] << ", ";
    // }
    // std::cout << "}" << std::endl;

    // std::cout << "irow = {";
    // for (int i=0; i<nnz; ++i) {
    //   std::cout << irow[i] << ", ";
    // }
    // std::cout << "}" << std::endl;

    // bool is_ok = check_data(n, pcol, irow, 'L');
    // std::cout << "is_ok=" << is_ok << std::endl;
  
    int n_eig = number_eigenvalues(scaled_data_.get_t(), 50);
    if (n_eig >= n) {
      n_eig = n-1;
    }
    // std::cout << "n_eig=" << n_eig << std::endl;
    ARluSymMatrix<double> matrix(n, nnz, A, irow, pcol, 'L');
    ARluSymStdEig<double> dprob(n_eig, matrix, 0.0);
    dprob.FindEigenvectors();
    // std::cout << "Order is " << order_ << std::endl;
    // std::cout << "T(2) is " << scaled_data_.get_t() << std::endl;

    delete [] irow;
    delete [] pcol;
    delete [] A;

    // std::cout << "upper_diag_elements.n_rows() = " << 
    //   upper_diag_elements.n_rows << std::endl;

    // arma::sp_mat system_matrix(locations, 
    // 			     system_matrix_, 
    // 			     np,np);
    // // system_matrix_.save("/home/gdinolov/Research/PDE-solvers/src/finite-difference/system_matrix.mat");
    // // locations.save("/home/gdinolov/Research/PDE-solvers/src/finite-difference/locations.mat");
    // // std::cout << "np=" << np << std::endl;

    arma::mat eigvec(np, n_eig);
    arma::vec eigval(n_eig);  
  
    for (int i=0; i<n_eig; ++i) {
      eigval(i) = dprob.Eigenvalue(i);
      for (int j=0; j<np; ++j) {
	eigvec(j,i) = dprob.Eigenvector(i,j);
      }
    }
  
    // for (unsigned i=0; i<locations.n_cols; ++i) {
    //   system_matrix( locations(0,i), locations(1,i) ) = 
    //     system_matrix_(i);
    // }
  
    // std::ofstream system_matrix_file;
    // system_matrix_file.open("/home/gdinolov/Research/PDE-solvers/src/finite-difference/system_matrix.csv");
    // // No header
    // for (int i=0; i<np; ++i) {
    //   for (int j=0; j<np; ++j) {
    //     if (j==np-1) {
    // 	system_matrix_file << system_matrix(i,j) << "\n";	
    //     } else {
    // 	system_matrix_file << system_matrix(i,j) << ",";
    //     }
    //   }
    // }
    // system_matrix_file.close();
    
    // eigval.resize(np);
    // eigvec.resize(np,np);
    // arma::eig_sym(eigval, eigvec, dense_system_matrix);
    // arma::vec exp_eigval(np);
    // for (int i=0; i<np; ++i) {
    //   exp_eigval(i) = exp(scaled_data_.get_t()*eigval(i));
    //   std::cout << "exp_eigval(" << i << ") = " << exp_eigval(i) << std::endl;
    // }

    arma::vec exp_eigval(n_eig);
    for (int i=0; i<n_eig; ++i) {
      exp_eigval(i) = exp(scaled_data_.get_t()*eigval(i));
      // std::cout << "exp_eigval(" << i << ") = " << exp_eigval(i) << std::endl;
    }
  
    unsigned IC_index = quantized_shifted_data_.get_i_0() - i_L - 1;
    unsigned solution_index = quantized_shifted_data_.get_i_T() - i_L - 1;
  
    // exp(Lambda * t) V^t IC_vector
    arma::vec pre_solution_vec =
      arma::diagmat(exp_eigval) * 
      eigvec.row(IC_index).t() * order_;
  
    // std::cout << "solution_index=" << solution_index << std::endl;
    // std::cout << "IC_index=" << IC_index << std::endl;
    // std::cout << "i_L=" << i_L << std::endl;
    double solution = dot(eigvec.row(solution_index), 
			  pre_solution_vec);
    return solution;
  }
}

int OneDHeatEquationFiniteDifferenceSolver::
number_eigenvalues(double t_2, double Delta_cut) const 
{
  arma::vec v = arma::ones(order_);

  for (int i=0; i<order_; ++i) {
    v(i) = square(i+1);
  }
  
  int n_eig = 0;
  double cutoff = 2.0*Delta_cut/(t_2 * square(pi()));
				
  for (int i=0; i<order_; ++i) {
    if (v(i) < cutoff) {
      n_eig++;
    }
  }
  return n_eig+1;
}

bool OneDHeatEquationFiniteDifferenceSolver::
check_data(int n, int* pcol, int* irow, char uplo) const 
{
  // std::cout << "In check_data()" << std::endl;
  int i, j, k;

  // Checking if pcol is in ascending order.

  i = 0;
  while ((i!=n)&&(pcol[i]<=pcol[i+1])) i++;
  if (i!=n) {
    std::cout << "i!=n" << std::endl;
    return false;
  }

  // Checking if irow components are in order and within bounds.

  for (i=0; i!=n; i++) {
    j = pcol[i];
    k = pcol[i+1]-1;
    if (j<=k) {
      if (uplo == 'U') {
        if ((irow[j]<0)||(irow[k]>i)) {
	  std::cout << "(irow[j]<0)||(irow[k]>i)" << std::endl;
	  return false;
	}
      }
      else { // uplo == 'L'.
        if ((irow[j]<i)||(irow[k]>=n)) {
	  std::cout << "i=" << i << std::endl;
	  std::cout << "k=" << k << std::endl;
	  std::cout << "k=pcol[i+1]-1=" << pcol[i+1]-1 << std::endl;
	  std::cout << "(irow[j]<i)||(irow[k]>=n)" << std::endl;
	  std::cout << "(irow[j]<i)=" << (irow[j]<i) << std::endl;
	  std::cout << "(irow[k]>=n)=" << (irow[k]>=n) << std::endl;
	  return false;
	}
      }
      while ((j!=k)&&(irow[j]<irow[j+1])) j++;
      if (j!=k) {
	std::cout << "j!=k" << std::endl;
	return false;
      }
    }
  }   

  return true;
}
