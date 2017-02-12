#include <algorithm>
#include <vector>
#include "PDEDataTypes.hpp"

PointInterpolation::PointInterpolation()
  : n_I_(0),
    i_m_(std::vector<int> (0)),
    j_m_(std::vector<int> (0)),
    q_m_(std::vector<double> (0)),
    case_("NA")
{}

PointInterpolation::PointInterpolation(int n_I,
				       std::vector<int> i_m,
				       std::vector<int> j_m,
				       std::vector<double> q_m,
				       std::string case_in_region)
  : n_I_(n_I),
    i_m_(i_m),
    j_m_(j_m),
    q_m_(q_m),
    case_(case_in_region)
{}

PointInterpolation::
PointInterpolation(double x,
		   double y,
		   int i_L,
		   int i_R,
		   int j_L,
		   int j_U,
		   int order)
{
  double h = 1.0/order;
  int i_zeta = std::floor((x+y)/(2.0*h));
  int i_nu = std::floor((y-x)/(2.0*h));
  double beta_zeta = (x+y)/(2.0*h) - i_zeta;
  double beta_nu = (y-x)/(2.0*h) - i_nu;

  int i_0 = i_zeta - i_nu;
  int j_0 = i_zeta + i_nu;

  if (j_0 == j_U) {
    // Case 1, only (i_0,j_0) within [i_L,i_R]x[j_L,j_U]
    case_ = "case1";
    n_I_ = 1;
    i_m_ = {i_0};
    j_m_ = {j_0};
    q_m_ = {1.0};
  } else if (j_0 == j_U-1 && i_0 == i_L) {
    // Case 2A, upper left corner
    case_ = "case2a";
    n_I_ = 2;
    i_m_ = {i_0, i_0+1};
    j_m_ = {j_0, j_0+1};
    q_m_ = {1-beta_zeta, beta_zeta};
  } else if (j_0 == j_U-1 && i_0 == i_R) {
    // Case 2B, upper right corner
    case_ = "case2b";
    n_I_ = 2;
    i_m_ = {i_0, i_0-1};
    j_m_ = {j_0, j_0+1};
    q_m_ = {1-beta_nu, beta_nu};
  } else if (j_0 == j_L-1 && i_0 == i_R) {
    // Case 2C, lower right corner
    case_ = "case2c";
    n_I_ = 2;
    i_m_ = {i_0-1, i_0};
    j_m_ = {j_0+1, j_0+2};
    q_m_ = {1-beta_zeta, beta_zeta};
  } else if (j_0 == j_L-1 && i_0 == i_L) {
    // Case 2D, cannot occur becase (i_L, j_L-1) is not a grid point.
    std::cout << "In quantize_data, cannot occur!!" << std::endl;
  } else if (j_0 == j_U-1 && i_L+1 <= i_0 && i_0 <= i_R-1) {
    // Case 3A, upper boundary, away from corners
    case_ = "case3a";
    n_I_ = 3;
    i_m_ = {i_0, i_0-1, i_0+1};
    j_m_ = {j_0, j_0+1, j_0+1};
    q_m_ = {1-(beta_zeta+beta_nu), beta_nu, beta_zeta};
  } else if (j_0 == j_L-1 && i_L+1 <= i_0 && i_0 <= i_R-1) {
    // Case 3B, lower boundary awa from corners
    case_ = "case3b";
    n_I_ = 3;
    i_m_ = {i_0-1, i_0+1, i_0};
    j_m_ = {j_0+1, j_0+1, j_0+2};
    q_m_ = {1-beta_zeta, 1-beta_nu, beta_zeta+beta_nu-1};
  } else if (i_0 == i_R && j_L <= j_0 && j_0 <= j_U-2) {
    // Case 3C, right boundary, away from two corners
    case_ = "case3c";
    n_I_ = 3;
    i_m_ = {i_0, i_0-1, i_0};
    j_m_ = {j_0, j_0+1, j_0+2};
    q_m_ = {1-beta_nu, beta_nu-beta_zeta, beta_zeta};
  } else if (i_0 == i_L && j_L <= j_0 && j_0 <= j_U-2) {
    // Case 3D, left boundary, away from corners
    case_ = "case3d";
    n_I_ = 3;
    i_m_ = {i_0, i_0+1, i_0};
    j_m_ = {j_0, j_0+1, j_0+2};
    q_m_ = {1-beta_zeta, beta_zeta-beta_nu, beta_nu};
  } else {
    case_ = "case4";
    n_I_ = 4;
    i_m_ = {i_0, i_0-1, i_0+1, i_0};
    j_m_ = {j_0, j_0+1, j_0+1, j_0+2};
    q_m_ = {(1-beta_zeta)*(1-beta_nu), 
	   (1-beta_zeta)*beta_nu, 
	   beta_zeta*(1-beta_nu),
	   beta_zeta*beta_nu};
  }
}

int PointInterpolation::get_n_I() const
{
  return n_I_;
}

std::vector<int> PointInterpolation::get_i_m() const
{
  return i_m_;
}

std::vector<int> PointInterpolation::get_j_m() const
{
  return j_m_;
}

std::vector<double> PointInterpolation::get_q_m() const
{
  return q_m_;
}

std::string PointInterpolation::get_case() const
{
  return case_;
}

std::ostream& operator<<(std::ostream& output_stream, 
			 const PointInterpolation& point_interpolation) 
{
  output_stream << "n_I=" << point_interpolation.n_I_ << ",\n";
  output_stream << "case = " << point_interpolation.case_ << ",\n";
  output_stream << "(i_m, j_m, q_m) = " << ",\n";
  for (int i=0; i<point_interpolation.n_I_; ++i) {
    output_stream << point_interpolation.i_m_[i] << ", " 
		  << point_interpolation.j_m_[i] << ", "
		  << point_interpolation.q_m_[i] << "\n";
  }
  return output_stream;
}


ContinuousProblemData::ContinuousProblemData()
  : x_T_(0.5),
    y_T_(0.5),
    x_0_(0.5),
    y_0_(0.5),
    t_(1.0),
    a_(0.0),
    b_(1.0),
    c_(0.0),
    d_(1.0)
{}

ContinuousProblemData::ContinuousProblemData(double x_T,
					     double y_T,
					     double x_0,
					     double y_0,
					     double t,
					     double a,
					     double b,
					     double c,
					     double d)
  : x_T_(x_T),
    y_T_(y_T),
    x_0_(x_0),
    y_0_(y_0),
    t_(t),
    a_(a),
    b_(b),
    c_(c),
    d_(d)
{}

double ContinuousProblemData::get_x_T() const 
{
  return x_T_;
}
void ContinuousProblemData::set_x_T(double x_T) 
{
  x_T_ = x_T;
}
double ContinuousProblemData::get_y_T() const
{
  return y_T_;
}
void ContinuousProblemData::set_y_T(double y_T)
{
  y_T_ = y_T;
}
double ContinuousProblemData::get_x_0() const 
{
  return x_0_;
}
double ContinuousProblemData::get_y_0() const
{
  return y_0_;
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
double ContinuousProblemData::get_c() const
{
  return c_;
}
double ContinuousProblemData::get_d() const
{
  return d_;
}
void ContinuousProblemData::switch_x_y()
{
  double new_a = c_;
  double new_b = d_;
  double new_x_T = y_T_;
  double new_x_0 = y_0_;
  
  c_ = a_;
  d_ = b_;
  a_ = new_a;
  b_ = new_b;
  y_T_ = x_T_;
  y_0_ = x_0_;
  x_T_ = new_x_T;
  x_0_ = new_x_0;
}
std::ostream& operator<<(std::ostream& output_stream, 
			  const ContinuousProblemData& c_prod_data) 
{
  output_stream << "t=" << c_prod_data.t_ << "\n";
  output_stream << "x_T=" << c_prod_data.x_T_ << ", " << "y_T=" << c_prod_data.y_T_ << "\n";
  output_stream << "x_0=" << c_prod_data.x_0_ << ", " << "y_0=" << c_prod_data.y_0_ << "\n";
  output_stream << "(a,b,c,d)=" << "(" << 
    c_prod_data.a_ << "," <<
    c_prod_data.b_ << "," <<
    c_prod_data.c_ << "," <<
    c_prod_data.d_ << ")";
  return output_stream;
}

DiscreteProblemData::DiscreteProblemData()
  : initial_condition_(PointInterpolation()),
    final_condition_(PointInterpolation())
{}

DiscreteProblemData::DiscreteProblemData(PointInterpolation initial_condition,
					 PointInterpolation final_condition)
  : initial_condition_(initial_condition),
    final_condition_(final_condition)
{}

const PointInterpolation & DiscreteProblemData::get_initial_condition() const
{
  return initial_condition_;
}

const PointInterpolation & DiscreteProblemData::get_final_condition() const
{
  return final_condition_;
}

std::ostream& operator<<(std::ostream& output_stream, 
			  const DiscreteProblemData& c_prod_data) 
{
  output_stream << "initial_condition=" << c_prod_data.initial_condition_ << "\n";
  output_stream << "final_condition=" << c_prod_data.final_condition_;
  return output_stream;
}


QuantizedProblemData::QuantizedProblemData()
  : i_T_(0),
    j_T_(0),
    i_0_(0),
    j_0_(0),
    i_L_(0),
    i_R_(1),
    j_U_(0),
    j_L_(1)
{}

QuantizedProblemData::QuantizedProblemData(int i_T, 
					   int j_T,
					   int i_0,
					   int j_0,
					   int i_L, 
					   int i_R,
					   int j_L, 
					   int j_U)
  : i_T_(i_T),
    j_T_(j_T),
    i_0_(i_0),
    j_0_(j_0),
    i_L_(i_L),
    i_R_(i_R),
    j_U_(j_U),
    j_L_(j_L)
{}

int QuantizedProblemData::get_i_T() const {
  return i_T_;
}
int QuantizedProblemData::get_j_T() const {
  return j_T_;
}
int QuantizedProblemData::get_i_0() const {
  return i_0_;
}
int QuantizedProblemData::get_j_0() const {
  return j_0_;
}
int QuantizedProblemData::get_i_L() const {
  return i_L_;
}
int QuantizedProblemData::get_i_R() const {
  return i_R_;
}
int QuantizedProblemData::get_j_L() const {
  return j_L_;
}
int QuantizedProblemData::get_j_U() const {
  return j_U_;
}

std::ostream& operator<<(std::ostream& output_stream, 
			  const QuantizedProblemData& q_prod_data) 
{
  output_stream << "i_T=" << q_prod_data.i_T_ << ", " << "j_T=" << q_prod_data.j_T_ << "\n";
  output_stream << "i_0=" << q_prod_data.i_0_ << ", " << "j_0=" << q_prod_data.j_0_ << "\n";
  output_stream << "(i_L,i_R,j_L,j_U)=" << "(" << 
    q_prod_data.i_L_ << "," <<
    q_prod_data.i_R_ << "," <<
    q_prod_data.j_L_ << "," <<
    q_prod_data.j_U_ << ")";
  return output_stream;
}

BoundaryIndeces::BoundaryIndeces()
  : i_L_(0),
    i_R_(0),
    j_L_(0),
    j_U_(0)
{}

BoundaryIndeces::BoundaryIndeces(int i_L,
				 int i_R,
				 int j_L,
				 int j_U)
  : i_L_(i_L),
    i_R_(i_R),
    j_L_(j_L),
    j_U_(j_U)
{}

int BoundaryIndeces::get_i_L() const 
{
  return i_L_;
}

int BoundaryIndeces::get_i_R() const 
{
  return i_R_;
}

int BoundaryIndeces::get_j_L() const
{
  return j_L_;
}

int BoundaryIndeces::get_j_U() const 
{
  return j_U_;
}

std::ostream& operator<<(std::ostream& output_stream, 
			 const BoundaryIndeces& boundary_indeces) 
{
  output_stream << "i_L=" << boundary_indeces.i_L_ << ", " 
		<< "i_R=" << boundary_indeces.i_R_ << ",";
  output_stream << "j_L=" << boundary_indeces.j_L_ << ", " 
		<< "j_U=" << boundary_indeces.j_U_ << "\n";
  return output_stream;
}

EigenproblemIndexing::EigenproblemIndexing(unsigned i_L, 
					   unsigned i_R, 
					   unsigned j_L, 
					   unsigned j_U)
{
  arma::uvec ja( (j_U-1)-(j_L+1)+1 );
  for (unsigned i=0; i<ja.n_rows; ++i) {
    ja(i) = j_L + 1 + i;
  }
  // how many rows are there in the quantized region
  int j_num = ja.n_rows;
  
  // for each row, we'll count the number of points and put them in
  // this vector.
  arma::Col<arma::uword> i_num(j_num);

  // the index of i starts at either of two possible places and we'll
  // keep track of that in this container.
  i_beg_ = arma::Col<arma::uword> (j_num);

  for (unsigned j=0; j<ja.n_rows; ++j) {
    i_beg_(j) = i_L + 1 + (i_L+ja(j)+1)%2;
    i_num(j) = std::floor(((i_R-1)-i_beg_(j))/2.0)+1;
  }

  arma::Col<arma::uword> i_num_cumsum = cumsum(i_num);
  i_cum_ = arma::Col<arma::uword> (i_num.n_rows+1);
  i_cum_(0) = 0;
  for (unsigned i=0; i<i_num_cumsum.n_rows; ++i) {
    i_cum_(i+1) = i_num_cumsum(i);
  }
}

const arma::Col<arma::uword> EigenproblemIndexing::get_i_cum() const
{
  return i_cum_;
}

const arma::Col<arma::uword> EigenproblemIndexing::get_i_beg() const
{
  return i_beg_;
}

SystemMatrices::SystemMatrices(arma::vec system_matrix,
			       arma::umat index_matrix)
  : system_matrix_(system_matrix),
    index_matrix_(index_matrix)
{}

const arma::vec & SystemMatrices::get_system_matrix() const
{
  return system_matrix_;
}

const arma::umat & SystemMatrices::get_index_matrix() const
{
  return index_matrix_;
}
