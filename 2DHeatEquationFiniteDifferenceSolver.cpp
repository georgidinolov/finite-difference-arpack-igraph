// This is the implementation for the finite difference method for
// solving the heat equation on a bounded domain with absorbing
// boundaries
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
extern "C" {
#include "igraph.h"
#include "igraph_sparsemat.h"
}

#include <algorithm>
#include <armadillo>
#include <cstddef>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <omp.h>
#include <vector>
#include "2DHeatEquationFiniteDifferenceSolver.hpp"

using namespace arma;

namespace{ 
  inline double square(double x) {
    return x * x;
  }
  
  inline double pi() {
    return std::atan(1)*4;
  }

  inline double round_delta(double input, double delta) {
    return delta * std::round(input/delta);
  }
}

TwoDHeatEquationFiniteDifferenceSolver::TwoDHeatEquationFiniteDifferenceSolver()
  : original_data_(0.5,
		   0.5,
		   0.5,
		   0.5,
		   1.0,
		   0.0,
		   1.0,
		   0.0,
		   1.0),
    eigenvectors_(24),
    eigenvalues_(24),
    order_(64),
    rho_(0.0),
    sigma_x_(1.0),
    sigma_y_(1.0),
    correction_factor_(1.0),
    switch_x_y_(false),
    system_matrix_one_(arma::Mat<double>(1,1)),
    system_matrix_two_(arma::Mat<double>(1,1)),
    index_matrix_(arma::Mat<arma::uword>(1,1)),
    system_matrix_(arma::vec(1))
{
  eigenprobs_ = std::vector<Eigenproblem *> ();
  for (int i=0; i<24; ++i) {
    Eigenproblem * new_prob = new Eigenproblem();
    eigenprobs_.push_back(new_prob);
  }
  
  scale_data();
  quantize_data();

  if (boundary_indeces_.get_j_U() < 3) {
    order_ = 2*64;
    scale_data();
    quantize_data();
  }
  
  pre_calc_S_matrices();
}

TwoDHeatEquationFiniteDifferenceSolver::
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
				       double t)
  : original_data_(x_t,y_t,0,0,t,a,b,c,d),
    eigenvectors_(24),
    eigenvalues_(24),
    order_(order),
    rho_(rho),
    sigma_x_(sigma_x),
    sigma_y_(sigma_y),
    correction_factor_(1.0),
    eigenprobs_(24)
{
  // TODO(gdinolov) : I need to throw an exception if (x,y) is outside
  // the boundary.
  set_switch_x_y();
  system_matrix_one_ = arma::Mat<double>(1,1);
  system_matrix_two_ = arma::Mat<double>(1,1);
  index_matrix_ = arma::Mat<arma::uword>(1,1);
  system_matrix_ = arma::vec(1);

  // Steps 1 and 2 in the write-up.
  scale_data();
  // Step 3 in the write-up
  quantize_data();

  while (boundary_indeces_.get_j_U() < 3) {
    order_ = 2*order_;
    scale_data();
    quantize_data();
  }
  
  pre_calc_S_matrices();
}

TwoDHeatEquationFiniteDifferenceSolver::~TwoDHeatEquationFiniteDifferenceSolver()
{
  for (int index=0; index<24; ++index) {
    if (index != 10 &&
	index != 9 &&
	index != 18 &&
	index != 17) {
      delete eigenprobs_[index];
    }
    eigenprobs_[index] = nullptr;
  }
}

// const ContinuousProblemData& TwoDHeatEquationFiniteDifferenceSolver::
// get_quantized_continuous_data() const 
// {
//   return quantized_continuous_data_;
// }

// const DiscreteProblemData& TwoDHeatEquationFiniteDifferenceSolver::
// get_quantized_discrete_data() const 
// {
//   return quantized_discrete_data_;
// }

// double TwoDHeatEquationFiniteDifferenceSolver::get_sigma_x() const 
// {
//   return sigma_x_;
// }

// double TwoDHeatEquationFiniteDifferenceSolver::get_sigma_y() const 
// {
//   return sigma_y_;
// }

const BoundaryIndeces& TwoDHeatEquationFiniteDifferenceSolver::
get_boundary_indeces() const
{
  return boundary_indeces_;
}

void TwoDHeatEquationFiniteDifferenceSolver::save_data_point() const
{
  std::ostringstream name;
  name << order_ << "-"
       << sigma_x_ << "-"
       << sigma_y_ << "-"
       << rho_ << "-"
       << original_data_.get_a() << "-"
       << original_data_.get_b() << "-"
       << original_data_.get_c() << "-"
       << original_data_.get_d();
  std::string output_file_name = "bad-data-points/bad-data-point-" + name.str() + ".txt";
  std::ofstream output_file;
  output_file.open(output_file_name);
  output_file << "order,sigma_x,sigma_y,rho,x_0,y_0,a_x,x_T,b_x,a_y,y_T,b_y\n";
  output_file << order_ << ","
	      << sigma_x_ << ","
	      << sigma_y_ << ","
	      << rho_ << ","
	      << original_data_.get_x_0() << ","
    	      << original_data_.get_y_0() << ","
	      << original_data_.get_a() << ","
	      << original_data_.get_x_T() << ","
    	      << original_data_.get_b() << ","
    	      << original_data_.get_c() << ","
    	      << original_data_.get_y_T() << ","
    	      << original_data_.get_d() << "\n";
  output_file.close();
}

void TwoDHeatEquationFiniteDifferenceSolver::set_switch_x_y()
{
  if ( (original_data_.get_b() - original_data_.get_a())/
       (sigma_x_/sqrt(2.0)) >= 
       (original_data_.get_d() - original_data_.get_c())/
       (sigma_y_/sqrt(2.0)) ) {
    switch_x_y_ = false;
  } else {
    switch_x_y_ = true;
  }
}

void TwoDHeatEquationFiniteDifferenceSolver::set_sigma_x(double sigma_x)
{
  sigma_x_ = sigma_x;
  set_switch_x_y();
    // Steps 1 and 2 in the write-up.
  scale_data();
  // Step 3 in the write-up
  quantize_data();
  pre_calc_S_matrices();
}

void TwoDHeatEquationFiniteDifferenceSolver::set_sigma_y(double sigma_y)
{
  sigma_y_ = sigma_y;
  set_switch_x_y();
    // Steps 1 and 2 in the write-up.
  scale_data();
  // Step 3 in the write-up
  quantize_data();
  pre_calc_S_matrices();
}

void TwoDHeatEquationFiniteDifferenceSolver::set_order(int order)
{
  order_ = order;
  quantize_data();
  pre_calc_S_matrices();
}

double TwoDHeatEquationFiniteDifferenceSolver::solve() 
{
  const Eigenproblem * eigenproblem_ptr = 
    solve_eigenproblem(boundary_indeces_.get_i_L(),
		       boundary_indeces_.get_i_R(),
		       boundary_indeces_.get_j_L(),
		       boundary_indeces_.get_j_U());

  double solution = solve_discretized_PDE(boundary_indeces_.get_i_L(),
  					  boundary_indeces_.get_i_R(),
  					  boundary_indeces_.get_j_L(),
  					  boundary_indeces_.get_j_U(),
  					  eigenproblem_ptr);
  delete eigenproblem_ptr;
  return solution * correction_factor_;
}

double TwoDHeatEquationFiniteDifferenceSolver::solve(double x_T, 
						     double y_T, 
						     double T) 
{
  // TODO(georgid): We need to check if the new point is in between
  // the boundaries
  original_data_.set_x_T(x_T);
  original_data_.set_y_T(y_T);
  original_data_.set_t(T);
  scale_data();
  quantize_data();
  pre_calc_S_matrices();
  
  double solution = solve();
  return solution;
}

double TwoDHeatEquationFiniteDifferenceSolver::likelihood() 
{
  std::vector<double> solutions = std::vector<double> (24);
  std::vector<const Eigenproblem*> eigenproblem_ptr_vector = 
    std::vector<const Eigenproblem*> (24);
  std::vector<int> k1s (24);
  std::vector<int> k2s (24);
  std::vector<int> k3s (24);
  std::vector<int> k4s (24);

  for (int k1=0; k1<2; ++k1) {
    for (int k2=0; k2<2; ++k2) {
      for (int k3=0; k3<2; ++k3) {
  	for (int k4=0; k4<3; ++k4) {
  	  int index = k1 + 2*k2 + 4*k3 + 8*k4;
	  k1s[index] = k1;
	  k2s[index] = k2;
	  k3s[index] = k3;
	  k4s[index] = k4;

  	  std::vector<int> indeces;
  	  if (index != 10 &&
  	      index != 9 &&
  	      index != 18 &&
  	      index != 17) {
  	    if (index == 5) {
  	      indeces = {index, 10};
  	    } else if (index == 6) {
  	      indeces = {index, 9};
  	    } else if (index == 13) {
  	      indeces = {index, 18};
  	    } else if (index == 14) {
  	      indeces = {index, 17};
  	    } else {
  	      indeces = {index};
  	    }

	    const Eigenproblem* eigenproblem_ptr =
	      solve_eigenproblem(boundary_indeces_.get_i_L()-k1,
				 boundary_indeces_.get_i_R()+k2,
				 boundary_indeces_.get_j_L()-k3,
				 boundary_indeces_.get_j_U()+k4);
	    
	    for (unsigned i=0; i<indeces.size(); ++i) {
	      delete eigenproblem_ptr_vector[indeces[i]];
	      eigenproblem_ptr_vector[indeces[i]] = eigenproblem_ptr;
	    }

  	  }
  	}
      }
    }
  } 

  double h4 = square(square(1.0/order_));  
  int i;
  for (i=0; i<24; ++i) {
    solutions[i] = solve_discretized_PDE(boundary_indeces_.get_i_L()-k1s[i],
					 boundary_indeces_.get_i_R()+k2s[i],
					 boundary_indeces_.get_j_L()-k3s[i],
					 boundary_indeces_.get_j_U()+k4s[i],
					 eigenproblem_ptr_vector[i]);
  } 

  std::vector<double> V_0 = {1.0,-1.0,-1.0,1.0,-1.0,1.0,1.0,-1.0};
  std::vector<double> V_ef = std::vector<double> (3*V_0.size());

  for (int i=0; i<3; ++i) {
    for (int j=0; j<V_0.size(); ++j) {
      if (i==0) {
	V_ef[i*V_0.size() + j] = alpha_*V_0[j];
      } else if (i==1) {
	V_ef[i*V_0.size() + j] = -(2*alpha_-1)*V_0[j];
      } else {
	V_ef[i*V_0.size() + j] = -(1-alpha_)*V_0[j];
      }
    }
  }


  double derivative = 0;
  for (int i=0; i<24; ++i) {
    derivative = derivative + V_ef[i]*solutions[i];
  }

  // Deleting the eigenproblems objects allocated on the heap.
  for (int k1=0; k1<2; ++k1) {
    for (int k2=0; k2<2; ++k2) {
      for (int k3=0; k3<2; ++k3) {
  	for (int k4=0; k4<3; ++k4) {
  	  int index = k1 + 2*k2 + 4*k3 + 8*k4;
  	  std::vector<int> indeces;
  	  if (index != 10 &&
  	      index != 9 &&
  	      index != 18 &&
  	      index != 17) {
	    
	    delete eigenproblem_ptr_vector[index];
  	  }
  	}
      }
    }
  }

  return correction_factor_*derivative/h4;
}

void TwoDHeatEquationFiniteDifferenceSolver::scale_data() {
  double sigma_x = sigma_x_;
  double sigma_y = sigma_y_;
  if (switch_x_y_) {
    original_data_.switch_x_y();

    double new_sigma_x = sigma_y;
    sigma_y = sigma_x_ / sqrt(2);
    sigma_x = new_sigma_x / sqrt(2);
  } else {
    sigma_x = sigma_x / sqrt(2);
    sigma_y = sigma_y / sqrt(2);
  }

  double scaled_x_T_1 = original_data_.get_x_T() / sigma_x;
  double scaled_x_0_1 = original_data_.get_x_0() / sigma_x;
  double scaled_a_1 = original_data_.get_a() / sigma_x;
  double scaled_b_1 = original_data_.get_b() / sigma_x;

  double scaled_y_T_1 = original_data_.get_y_T() / sigma_y;
  double scaled_y_0_1 = original_data_.get_y_0() / sigma_y;
  double scaled_c_1 = original_data_.get_c() / sigma_y;
  double scaled_d_1 = original_data_.get_d() / sigma_y;

  double scaled_x_T = scaled_x_T_1 / 
    (scaled_b_1 - scaled_a_1);
  double scaled_x_0 = scaled_x_0_1 / 
    (scaled_b_1 - scaled_a_1);
  double scaled_a = scaled_a_1 / 
    (scaled_b_1 - scaled_a_1);
  double scaled_b = scaled_b_1 / 
    (scaled_b_1 - scaled_a_1);
  double scaled_y_T = scaled_y_T_1 / 
    (scaled_b_1 - scaled_a_1);
  double scaled_y_0 = scaled_y_0_1 / 
    (scaled_b_1 - scaled_a_1);
  double scaled_c = scaled_c_1 / 
    (scaled_b_1 - scaled_a_1);
  double scaled_d = scaled_d_1 / 
    (scaled_b_1 - scaled_a_1);

  double scaled_T = original_data_.get_t() /
    square(scaled_b_1 - scaled_a_1);

  // std::cout << "scaled_b - scaled_a = " 
  // 	    << scaled_b - scaled_a << std::endl;

  // std::cout << "scaled_d - scaled_c = " 
  // 	    << scaled_d - scaled_c << std::endl;

  correction_factor_ = 1.0 / 
    (pow(scaled_b_1 - scaled_a_1, 
	 6)*
     pow(sigma_x_/sqrt(2), 3)*
     pow(sigma_y_/sqrt(2), 3));

  // std::cout << "correction_factor_ = " 
  // 	    << correction_factor_ << std::endl;

  scaled_data_ = ContinuousProblemData(scaled_x_T,
				       scaled_y_T,
				       scaled_x_0,
				       scaled_y_0,
				       scaled_T,
				       scaled_a,
				       scaled_b,
				       scaled_c,
				       scaled_d);
}

void TwoDHeatEquationFiniteDifferenceSolver::quantize_data()
{
  double h = 1.0/order_;

  // (x_0,y_0) is the location of the IC in the new coordinate system
  double x_0 = -1.0*(scaled_data_.get_a() - h/2.0);
  double y_0 = -1.0*(scaled_data_.get_c() - h/2.0);

  // a = scaled_data_.get_a() + x_0 = h/2
  // b = scaled_data_.get_b() + x_0 = 1 + h/2
  // a and b are the left and right boundaries of the new coordinate
  // system.
  double a = scaled_data_.get_a() + x_0;
  double b = scaled_data_.get_b() + x_0;

  // c = scaled_data_.get_c() + y_0 = h/2
  // d = scaled_data_.get_d() + y_0
  // c and d are the lower and upper boundaries of the new coordiante
  // system.
  double c = scaled_data_.get_c() + y_0;
  double d = scaled_data_.get_d() + y_0;

  double x_T = scaled_data_.get_x_T() + x_0;
  double y_T = scaled_data_.get_y_T() + y_0;

  int i_L = 1;
  int i_R = order_;
  int j_L = 1;
  int j_U = std::floor((d-0.5*h)/h);
  double alpha = (j_U + 1.5) - d/h;

  std::vector<double> x_0_i_L_h = {x_0, i_L*h};
  double max_x_0_i_L_h = *std::max_element(std::begin(x_0_i_L_h),
					   std::end(x_0_i_L_h));
  std::vector<double> max_x_0_i_L_h_i_R_h = {max_x_0_i_L_h, i_R*h};
  x_0 = *std::min_element(std::begin(max_x_0_i_L_h_i_R_h),
			  std::end(max_x_0_i_L_h_i_R_h));

  std::vector<double> x_T_i_L_h = {x_T, i_L*h};
  double max_x_T_i_L_h = *std::max_element(std::begin(x_T_i_L_h),
					   std::end(x_T_i_L_h));
  std::vector<double> max_x_T_i_L_h_i_R_h = {max_x_T_i_L_h, i_R*h};
  x_T = *std::min_element(std::begin(max_x_T_i_L_h_i_R_h),
			  std::end(max_x_T_i_L_h_i_R_h));
  
  std::vector<double> y0jLh = {y_0, j_L*h};
  std::vector<double> yTjLh = {y_T, j_L*h};

  y_0 = *std::max_element(std::begin(y0jLh),
			  std::end(y0jLh));
  y_T = *std::max_element(std::begin(yTjLh),
			  std::end(yTjLh));

  
  std::vector<double> y0diff_dist_1 = {y_0-j_U*h,0};
  std::vector<double> yTdiff_dist_1 = {y_T-j_U*h,0};
  double dist_1 = 
    *std::max_element(std::begin(y0diff_dist_1),
		      std::end(y0diff_dist_1)) + 
    *std::max_element(std::begin(yTdiff_dist_1),
		      std::end(yTdiff_dist_1));
  
  std::vector<double> y0diff_dist_2 = {y_0-(j_U+1)*h, 0};
  std::vector<double> yTdiff_dist_2 = {y_T-(j_U+1)*h, 0};
  double dist_2 = 
    (j_U + 1.5)*h - d + 
    *std::max_element(std::begin(y0diff_dist_2),
		      std::end(y0diff_dist_2)) + 
    *std::max_element(std::begin(yTdiff_dist_2),
		      std::end(yTdiff_dist_2));

  if (dist_2 <= dist_1) {
    j_U = j_U + 1;
    d = (j_U + 0.5)*h;
    alpha = (j_U + 1.5) - d/h;

    std::vector<double> y0jUh = {y_0, j_U*h};
    std::vector<double> yTjUh = {y_T, j_U*h};

    y_0 = *std::min_element(std::begin(y0jUh),
			    std::end(y0jUh));
    y_T = *std::min_element(std::begin(yTjUh),
			    std::end(yTjUh));
  } else {
    std::vector<double> y0jUh = {y_0, j_U*h};
    std::vector<double> yTjUh = {y_T, j_U*h};

    y_0 = *std::min_element(std::begin(y0jUh),
			    std::end(y0jUh));
    y_T = *std::min_element(std::begin(yTjUh),
			    std::end(yTjUh));
  }

  quantized_continuous_data_ = ContinuousProblemData(x_T,
  						     y_T,
  						     x_0,
  						     y_0,
  						     scaled_data_.get_t(),
  						     a,
  						     b,
  						     c,
  						     d);

  PointInterpolation initial_condition = PointInterpolation(x_0,
							    y_0,
							    i_L,
							    i_R,
							    j_L,
							    j_U,
							    order_);
  
  PointInterpolation final_condition = PointInterpolation(x_T,
							  y_T,
							  i_L,
							  i_R,
							  j_L,
							  j_U,
							  order_);

  quantized_discrete_data_ = 
    DiscreteProblemData(initial_condition, final_condition);

  boundary_indeces_ = BoundaryIndeces(i_L, i_R, j_L, j_U);
  alpha_ = alpha;
}

void TwoDHeatEquationFiniteDifferenceSolver::pre_calc_S_matrices()
{
  unsigned n_r = (order_+2)/2;
  int N = 2 * square(n_r);
  index_matrix_.resize(5*N,4);
  system_matrix_one_.resize(5*N,1);
  system_matrix_two_.resize(5*N,1);
  int k_0 = 0;
  
  for (unsigned j=1; j<=2*n_r; ++j) {
    for (unsigned i=1+(j-1)%2; i<=2*n_r; i=i+2) {
      index_matrix_.row(k_0) = arma::Row<arma::uword> {i,j,i,j};
      index_matrix_.row(k_0+1) = arma::Row<arma::uword> {i,j,i-1,j-1};
      index_matrix_.row(k_0+2) = arma::Row<arma::uword> {i,j,i+1,j+1};
      index_matrix_.row(k_0+3) = arma::Row<arma::uword> {i,j,i+1,j-1};
      index_matrix_.row(k_0+4) = arma::Row<arma::uword> {i,j,i-1,j+1};
      
      system_matrix_one_.row(k_0) = -2.0;
      system_matrix_one_.row(k_0+1) = 1.0;
      system_matrix_one_.row(k_0+2) = 1.0;
      system_matrix_one_.row(k_0+3) = 0.0;
      system_matrix_one_.row(k_0+4) = 0.0;

      system_matrix_two_.row(k_0) = -2.0;
      system_matrix_two_.row(k_0+1) = 0.0;
      system_matrix_two_.row(k_0+2) = 0.0;
      system_matrix_two_.row(k_0+3) = 1.0;
      system_matrix_two_.row(k_0+4) = 1.0;

      k_0 = k_0+5;
    }
  }
}

const SystemMatrices TwoDHeatEquationFiniteDifferenceSolver::
pre_calc_system_matrix(int i_L, int i_R, int j_L, int j_U) const
{
  double h = 1.0/order_;
  double h2 = 2.0 * h * h;

  arma::uvec ind = 
    find(index_matrix_.col(0) > i_L && 
	 index_matrix_.col(0) < i_R && 
	 index_matrix_.col(2) > i_L && 
	 index_matrix_.col(2) < i_R && 
	 //
	 index_matrix_.col(1) > j_L && 
	 index_matrix_.col(1) < j_U && 
	 index_matrix_.col(3) > j_L && 
	 index_matrix_.col(3) < j_U);

  arma::uvec all_columns = {0,1,2,3};
  arma::umat index_matrix = index_matrix_.elem(ind,all_columns);
  arma::vec system_matrix= 
    ((1.0+rho_)/h2) * system_matrix_one_.elem(ind) + 
    ((1.0-rho_)/h2) * system_matrix_two_.elem(ind);
  return SystemMatrices(system_matrix, index_matrix);
}

double TwoDHeatEquationFiniteDifferenceSolver::
solve_discretized_PDE(unsigned i_L, 
		      unsigned i_R, 
		      unsigned j_L, 
		      unsigned j_U,
		      const Eigenproblem * eigenproblem_ptr) const
{
  EigenproblemIndexing i_cum_i_beg = EigenproblemIndexing(i_L,
  							  i_R,
  							  j_L,
  							  j_U);

  arma::Col<arma::uword> i_beg = i_cum_i_beg.get_i_beg();
  arma::Col<arma::uword> i_cum = i_cum_i_beg.get_i_cum();

  int np = i_cum(i_cum.n_rows-1);
  int n_eig = number_eigenvalues(scaled_data_.get_t(), 36);
  if (n_eig >= np) {
    n_eig = np-1;
  }
  
  arma::mat eigvec(np,n_eig);
  arma::vec eigval(n_eig);

  for (int i=0; i<n_eig; ++i) {
    eigval(i) = eigenproblem_ptr->get_eigenvalue(i);
    for (int j=0; j<np; ++j) {
      eigvec(j,i) = eigenproblem_ptr->get_eigenvector_elem(j,i);
    }
  }

  arma::vec exp_eigval(n_eig);
  for (int i=0; i<n_eig; ++i) {
    exp_eigval(i) = exp(scaled_data_.get_t()*eigval(i));
  }

  const PointInterpolation IC = quantized_discrete_data_.get_initial_condition();
  const PointInterpolation FC = quantized_discrete_data_.get_final_condition();

  double h = 1.0/order_;
  double h2 = (2.0 * h * h);
  int n_I = IC.get_n_I();
  std::vector<int> i0_m = IC.get_i_m();
  std::vector<int> j0_m = IC.get_j_m();
  std::vector<double> q0_m = IC.get_q_m();
  std::string case_type = IC.get_case();
  arma::Col<double> q_multiplier = arma::zeros<arma::Col<double>>(eigvec.n_rows);
  // std::cout << "[" << i_L << "," << j_L << "] x " 
  // 	    << "[" << i_R << "," << j_U << "]" << std::endl;
    
  for (int i=0; i<n_I; ++i) {
    // std::cout << "(" << i0_m[i] << "," << j0_m[i] << ")" << std::endl;
    if (i0_m[i] > i_L &&
  	i0_m[i] < i_R &&
  	j0_m[i] > j_L &&
  	j0_m[i] < j_U) {
      unsigned IC_index = i_cum(j0_m[i]-j_L-1) +
  	(i0_m[i] - i_beg(j0_m[i]-j_L-1))/2;
      q_multiplier[IC_index] = q0_m[i];
    }
  }

  arma::Col<double> u_T = 
    eigvec * arma::diagmat(exp_eigval) *  eigvec.t() * q_multiplier;

  double solution = 0.0;
  int n_II = FC.get_n_I();
  std::vector<int> iT_m = FC.get_i_m();
  std::vector<int> jT_m = FC.get_j_m();
  std::vector<double> qT_m = FC.get_q_m();
  for (int i=0; i<n_II; ++i) {
    // std::cout << "(" << iT_m[i] << "," << jT_m[i] << ")" << std::endl;
    if (iT_m[i] > i_L &&
  	iT_m[i] < i_R &&
  	jT_m[i] > j_L &&
  	jT_m[i] < j_U) {
      unsigned solution_index = i_cum(jT_m[i]-j_L-1) +
  	(iT_m[i] - i_beg(jT_m[i]-j_L-1))/2;
      solution = solution + qT_m[i]*u_T[solution_index];
    }
  }
  return solution/h2;
}

int TwoDHeatEquationFiniteDifferenceSolver::
number_eigenvalues(double t_2, double Delta_cut) const 
{
  arma::mat A1 = arma::ones(order_, order_);
  arma::vec each_row = arma::vec(order_);

  for (int i=0; i<order_; ++i) {
    each_row(i) = square(i+1);
  }
  for (int i=0; i<order_; ++i) {
    A1.row(i) = each_row.t();
  }

  arma::mat v = arma::reshape((1-rho_)/2.0 * A1 + (1+rho_)/2.0 * A1.t(),
  			      square(order_), 1);
  int n_eig = 0;
  double cutoff = 1 + Delta_cut/(2.0*square(pi())*t_2);

  for (int i=0; i<square(order_); ++i) {
    if (v(i) < cutoff) {
      n_eig++;
    }
  }
  return n_eig;
}

// bool TwoDHeatEquationFiniteDifferenceSolver::
// check_data(int n, int* pcol, int* irow, char uplo) const 
// {
//   std::cout << "In check_data()" << std::endl;
//   int i, j, k;

//   // Checking if pcol is in ascending order.

//   i = 0;
//   while ((i!=n)&&(pcol[i]<=pcol[i+1])) i++;
//   if (i!=n) {
//     std::cout << "i!=n" << std::endl;
//     return false;
//   }

//   // Checking if irow components are in order and within bounds.

//   for (i=0; i!=n; i++) {
//     j = pcol[i];
//     k = pcol[i+1]-1;
//     if (j<=k) {
//       if (uplo == 'U') {
//         if ((irow[j]<0)||(irow[k]>i)) {
// 	  std::cout << "(irow[j]<0)||(irow[k]>i)" << std::endl;
// 	  return false;
// 	}
//       }
//       else { // uplo == 'L'.
//         if ((irow[j]<i)||(irow[k]>=n)) {
// 	  std::cout << "i=" << i << std::endl;
// 	  std::cout << "k=" << k << std::endl;
// 	  std::cout << "k=pcol[i+1]-1=" << pcol[i+1]-1 << std::endl;
// 	  std::cout << "(irow[j]<i)||(irow[k]>=n)" << std::endl;
// 	  std::cout << "(irow[j]<i)=" << (irow[j]<i) << std::endl;
// 	  std::cout << "(irow[k]>=n)=" << (irow[k]>=n) << std::endl;
// 	  return false;
// 	}
//       }
//       while ((j!=k)&&(irow[j]<irow[j+1])) j++;
//       if (j!=k) {
// 	std::cout << "j!=k" << std::endl;
// 	return false;
//       }
//     }
//   }   

//   return true;
// }

const Eigenproblem * TwoDHeatEquationFiniteDifferenceSolver::
solve_eigenproblem(unsigned i_L, 
		   unsigned i_R, 
		   unsigned j_L, 
		   unsigned j_U) const
{
  SystemMatrices system_matrix_index_matrix = 
    pre_calc_system_matrix(i_L, i_R, j_L, j_U);

  arma::vec system_matrix = system_matrix_index_matrix.get_system_matrix();
  arma::umat index_matrix = system_matrix_index_matrix.get_index_matrix();

  EigenproblemIndexing i_cum_i_beg = EigenproblemIndexing(i_L,
  							  i_R,
  							  j_L,
  							  j_U);
  arma::Col<arma::uword> i_cum = i_cum_i_beg.get_i_cum();
  arma::Col<arma::uword> i_beg = i_cum_i_beg.get_i_beg();

  int np = i_cum(i_cum.n_rows-1);
  
  // Locations are the indeces of the nonzero elements of the system
  // matrix.
  arma::Mat<arma::uword> locations(2,index_matrix.n_rows);

  locations.row(0) = (i_cum(index_matrix.col(1)-j_L-1) + 
		      (index_matrix.col(0) - i_beg(index_matrix.col(1)-j_L-1))/2).t();
  
  locations.row(1) = (i_cum(index_matrix.col(3)-j_L-1) + 
		      (index_matrix.col(2) - i_beg(index_matrix.col(3)-j_L-1))/2).t();
  arma::uvec upper_diag_elements = find(locations.row(1) >= locations.row(0));

  int n = np;
  int nnz = upper_diag_elements.n_rows;

  // NUMBER EIGENVALUES 
  int n_eig = number_eigenvalues(scaled_data_.get_t(), 36);
  if (n_eig >= np) {
    n_eig = np - 1;
  }

  // ########### IGRAPH START ###########
  igraph_matrix_t * vectors = new igraph_matrix_t;
  igraph_vector_t * values = new igraph_vector_t;
  igraph_arpack_options_t options;
  
  igraph_sparsemat_t A_igraph, B_igraph;
  igraph_sparsemat_init(&A_igraph, n, n, system_matrix.n_rows);

  for (unsigned i=0; i<upper_diag_elements.n_rows; ++i) {
    if (locations(0,upper_diag_elements(i)) ==
  	locations(1,upper_diag_elements(i))) {
      igraph_sparsemat_entry(&A_igraph,
  			     locations(0,upper_diag_elements(i)),
  			     locations(1,upper_diag_elements(i)),
  			     system_matrix(upper_diag_elements(i)));
    } else { 
      igraph_sparsemat_entry(&A_igraph,
  			     locations(0,upper_diag_elements(i)),
  			     locations(1,upper_diag_elements(i)),
  			     system_matrix(upper_diag_elements(i)));
      
      igraph_sparsemat_entry(&A_igraph,
  			     locations(1,upper_diag_elements(i)),
  			     locations(0,upper_diag_elements(i)),
  			     system_matrix(upper_diag_elements(i)));
    }
  }
  igraph_sparsemat_compress(&A_igraph, &B_igraph);
  igraph_sparsemat_destroy(&A_igraph);

  igraph_arpack_options_init(&options);
  options.n = n;
  options.nev = n_eig;
  options.ncv = 0;
  options.which[0] = 'L';
  options.which[1] = 'M';
  options.mode = 3;
  options.sigma = 1;
  options.tol = 1e-16;

  igraph_vector_init(values, options.nev);
  igraph_matrix_init(vectors, options.n, options.nev);

  igraph_set_error_handler(igraph_error_handler_printignore);
  int result = igraph_sparsemat_arpack_rssolve(&B_igraph,
					   &options,
					   /*storage=*/ 0,
					   values,
					   vectors,
					   IGRAPH_SPARSEMAT_SOLVE_LU);
  int counter = 0;
  while (result == IGRAPH_ARPACK_NOSHIFT && counter < 100) {
    counter = counter + 1;
    std::cout << "result = "
	      << result
	      << "; counter = "
	      << counter
	      << std::endl;
    result = igraph_sparsemat_arpack_rssolve(&B_igraph,
					   &options,
					   /*storage=*/ 0,
					   values,
					   vectors,
					   IGRAPH_SPARSEMAT_SOLVE_LU);
  }
  
  Eigenproblem * eigenproblem_ptr = new Eigenproblem(values,
						     vectors);
  igraph_sparsemat_destroy(&B_igraph);
  // IGRAPH END
    
  return eigenproblem_ptr;
}

// EIGENPROBLEM CLASS //
Eigenproblem::Eigenproblem()
{
  igraph_vector_t eigvals;
  igraph_matrix_t eigvecs;

  igraph_vector_init(&eigvals, 1);
  igraph_matrix_init(&eigvecs, 1, 1);

  eigenvalues_ptr_ = &eigvals;
  eigenvectors_ptr_ = &eigvecs;
}

Eigenproblem::Eigenproblem(igraph_vector_t * eigenvalues_ptr,
			   igraph_matrix_t * eigenvectors_ptr)
  : eigenvalues_ptr_(eigenvalues_ptr),
    eigenvectors_ptr_(eigenvectors_ptr)
{}

Eigenproblem::~Eigenproblem()
{
  igraph_vector_destroy(eigenvalues_ptr_);
  igraph_matrix_destroy(eigenvectors_ptr_);
}

const igraph_vector_t * Eigenproblem::get_eigenvalues_ptr() const
{
  return eigenvalues_ptr_;
}

const igraph_matrix_t * Eigenproblem::get_eigenvectors_ptr() const
{
  return eigenvectors_ptr_;
}

const double Eigenproblem::get_eigenvalue(long int i) const
{
  return igraph_vector_e(eigenvalues_ptr_, i);
}

const double Eigenproblem::get_eigenvector_elem(long int i,
						long int j) const
{
  return igraph_matrix_e(eigenvectors_ptr_, i, j);
}
