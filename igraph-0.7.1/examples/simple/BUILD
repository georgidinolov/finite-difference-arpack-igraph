cc_binary(
	name = "igraph_sparsemat",
	srcs = ["igraph_sparsemat.c"],
	includes = ["igraph.h", "igraph_sparsemat.h"],
	deps = ["//src/finite-difference-arpack-igraph/igraph-0.7.1:igraph"],
	copts = ["-Isrc/finite-difference-arpack-igraph/igraph-0.7.1/include"],
)

cc_binary(
	name = "igraph_sparsemat2",
	srcs = ["igraph_sparsemat2.c"],
	includes = ["igraph.h", "igraph_sparsemat.h"],
	deps = ["//src/finite-difference-arpack-igraph/igraph-0.7.1:igraph"],
	copts = ["-Isrc/finite-difference-arpack-igraph/igraph-0.7.1/include"],
)

cc_binary(
	name = "igraph_sparsemat5",
	srcs = ["igraph_sparsemat5.c"],
	includes = ["igraph.h", "igraph_sparsemat.h"],
	deps = ["//src/finite-difference-arpack-igraph/igraph-0.7.1:igraph"],
	copts = ["-Isrc/finite-difference-arpack-igraph/igraph-0.7.1/include"],
)

cc_binary(
	name = "igraph_eigen_matrix_symmetric_arpack",
	srcs = ["igraph_eigen_matrix_symmetric_arpack.c"],
	includes = ["igraph.h", "igraph_sparsemat.h"],
	deps = ["//src/finite-difference-arpack-igraph/igraph-0.7.1:igraph"],
	copts = ["-Isrc/finite-difference-arpack-igraph/igraph-0.7.1/include"],
)

cc_binary(
	name = "igraph_arpack_rnsolve",
	srcs = ["igraph_arpack_rnsolve.c"],
	includes = ["igraph.h"],
	deps = ["//src/finite-difference-arpack-igraph/igraph-0.7.1:igraph"],
	copts = ["-Isrc/finite-difference-arpack-igraph/igraph-0.7.1/include"],
)

cc_binary(
	name = "check-sparse-solver",
	srcs = ["check-sparse-solver.cpp"],
	includes = ["igraph.h", "igraph_sparsemat.h"],
	deps= ["//src/finite-difference-arpack-igraph/igraph-0.7.1:igraph"],
	copts = ["-Isrc/finite-difference-arpack-igraph/igraph-0.7.1/include",
	      	 "-fopenmp",
		 "-O"],
	linkopts = ["-fopenmp"],
)

