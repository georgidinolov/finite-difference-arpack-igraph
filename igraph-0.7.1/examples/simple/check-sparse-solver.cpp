extern "C" {
#include "igraph.h"
#include "igraph_sparsemat.h"
}
#include <chrono>
#include <omp.h>
#include <vector>
#include <iostream>

int main() {

  long int i;
  int j;
  int dimension = 2000;
  
  /***********************************************************************/
  int n_eigenproblems = 16;
  std::vector<igraph_sparsemat_t> system_mats_A (n_eigenproblems);
  std::vector<igraph_sparsemat_t> system_mats_B (n_eigenproblems);
  
  igraph_rng_seed(igraph_rng_default(), 42 * 42);

  for (j=0; j<n_eigenproblems; j++) {
    igraph_sparsemat_init(&system_mats_A[j], dimension, dimension,
  			  dimension + 2*(dimension-1));

    for (i=0; i<dimension; i++) {
      igraph_sparsemat_entry(&system_mats_A[j], i, i,
  			     igraph_rng_get_integer(igraph_rng_default(),
  						    -10, 10));
      if (i > 0) {
  	int off_diagonal_entry = igraph_rng_get_integer(igraph_rng_default(),
  							-10, 10);
  	igraph_sparsemat_entry(&system_mats_A[j], i-1, i, off_diagonal_entry);
  	igraph_sparsemat_entry(&system_mats_A[j], i, i-1, off_diagonal_entry);
      }
    }
    
    igraph_sparsemat_compress(&system_mats_A[j], &system_mats_B[j]);
    igraph_sparsemat_destroy(&system_mats_A[j]);
  }

    auto t1 = std::chrono::high_resolution_clock::now();
    omp_set_dynamic(0);
#pragma omp parallel for shared(system_mats_B) private(j)
  for (j=0; j<n_eigenproblems; j++) {
    igraph_matrix_t vectors;
    igraph_vector_t values;
    igraph_arpack_options_t options;
    
    igraph_arpack_options_init(&options);
    options.n = dimension;
    options.nev = 100;
    options.ncv = 0;
    options.which[0] = 'L';
    options.which[1] = 'M';
    options.mode = 1;
    options.sigma = 3;
    options.tol = 1e-16;
    
    igraph_vector_init(&values, options.nev);
    igraph_matrix_init(&vectors, options.n, 0);
    
    printf("options.which = %d on thread %d\n", j, omp_get_thread_num()); 
    
    igraph_arpack_storage_t storage;
    igraph_arpack_storage_init(&storage, dimension, dimension, dimension, true);
    
    
    igraph_sparsemat_arpack_rssolve(&system_mats_B[j], &options, &storage,
				    &values, &vectors,
				    IGRAPH_SPARSEMAT_SOLVE_LU);
    /* if (VECTOR(values)[0] != 1.0) { return 1; } */
    
    //igraph_vector_print(&values); 
    
    igraph_vector_destroy(&values);
    igraph_matrix_destroy(&vectors);
    igraph_arpack_storage_destroy(&storage);
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "duration = "
  	    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
  	    << " milliseconds\n";
  
  for (int j=0; j<n_eigenproblems; j++) {
    igraph_sparsemat_destroy(&system_mats_B[j]);
  }

  return 0;
}
