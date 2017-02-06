#include "armadillo"
#include <iostream>
#include <vector>
#include <string>

class PointInterpolation
{
public:
  PointInterpolation();
  PointInterpolation(int n_I,
		     std::vector<int> i_m,
		     std::vector<int> j_m,
		     std::vector<double> q_m,
		     std::string case_in_region);

  PointInterpolation(double x,
		     double y,
		     int i_L,
		     int i_R,
		     int j_L,
		     int j_U,
		     int order);

  int get_n_I() const;
  std::vector<int> get_i_m() const;
  std::vector<int> get_j_m() const;
  std::vector<double> get_q_m() const;
  std::string get_case() const;
  friend std::ostream& operator <<(std::ostream& output_stream, 
				   const PointInterpolation& point_interpolation);
private:
  int n_I_;
  std::vector<int> i_m_;
  std::vector<int> j_m_;
  std::vector<double> q_m_;
  std::string case_;
};

class ContinuousProblemData
{
public:
  ContinuousProblemData();
  ContinuousProblemData(double x_T,
			double y_T,
			double x_0,
			double y_0,
			double t,
			double a,
			double b,
			double c,
			double d);

  double get_x_T() const;
  void set_x_T(double x_T);

  double get_y_T() const;
  void set_y_T(double y_T);

  double get_x_0() const;
  double get_y_0() const;

  double get_t() const;
  void set_t(double t);

  double get_a() const;
  double get_b() const;
  double get_c() const;
  double get_d() const;
  void switch_x_y();
  friend std::ostream& operator <<(std::ostream& output_stream, 
				   const ContinuousProblemData& c_prod_data);

private:
  double x_T_;
  double y_T_;
  double x_0_;
  double y_0_;
  double t_;
  double a_;
  double b_;
  double c_;
  double d_;
};

class DiscreteProblemData
{
public:
  DiscreteProblemData();
  DiscreteProblemData(PointInterpolation initial_condition,
		      PointInterpolation final_condition);

  const PointInterpolation & get_initial_condition() const;
  const PointInterpolation & get_final_condition() const;
  friend std::ostream& operator <<(std::ostream& output_stream, 
				   const DiscreteProblemData& c_prod_data);

private:
  PointInterpolation initial_condition_;
  PointInterpolation final_condition_;
};

class QuantizedProblemData
{
public:
  QuantizedProblemData();
  QuantizedProblemData(int i_T, 
		       int j_T,
		       int i_0,
		       int j_0,
		       int i_L, 
		       int i_R,
		       int j_L, 
		       int j_U);

  int get_i_T() const;
  int get_j_T() const;
  int get_i_0() const;
  int get_j_0() const;
  int get_i_L() const;
  int get_i_R() const;
  int get_j_U() const;
  int get_j_L() const;
  friend std::ostream& operator <<(std::ostream& output_stream, 
				   const QuantizedProblemData& q_prod_data);
private:
  int i_T_;
  int j_T_;
  int i_0_;
  int j_0_;
  int i_L_;
  int i_R_;
  int j_U_;
  int j_L_;
};

class BoundaryIndeces
{
public:
  BoundaryIndeces();
  BoundaryIndeces(int i_L,
		  int i_R,
		  int j_L,
		  int j_U);
  int get_i_L() const;
  int get_i_R() const;
  int get_j_U() const;
  int get_j_L() const;
  friend std::ostream& operator <<(std::ostream& output_stream, 
				   const BoundaryIndeces& boundary_indeces);
private:
  int i_L_;
  int i_R_;
  int j_L_;
  int j_U_;
};

class EigenproblemIndexing
{
public:
  EigenproblemIndexing(unsigned i_L, 
		       unsigned i_R, 
		       unsigned j_L, 
		       unsigned j_U);
  const arma::Col<arma::uword> get_i_cum() const;
  const arma::Col<arma::uword> get_i_beg() const;

private:
  arma::Col<arma::uword> i_cum_;
  arma::Col<arma::uword> i_beg_;
};

class SystemMatrices
{
public:
  SystemMatrices(arma::vec system_matrix,
		 arma::umat index_matrix);

  const arma::vec & get_system_matrix() const;
  const arma::umat & get_index_matrix() const;

private:
  arma::vec system_matrix_;
  arma::umat index_matrix_;
};
