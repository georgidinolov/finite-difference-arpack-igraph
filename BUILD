cc_binary(
	name = "2d-finite-difference-test",
	srcs = ["2d-finite-difference-test.cpp"],
	includes = ["2DHeatEquationFiniteDifferenceSolver.hpp"],
	deps = [":2d-heat-equation-finite-difference",
	        "//src/brownian-motion:2d-brownian-motion"],
	copts = ["-Isrc/igraph-0.7.1/include"],
)

cc_binary(
	name = "1d-finite-difference-test",
	srcs = ["1d-finite-difference-test.cpp"],
	includes = ["1DHeatEquationFiniteDifferenceSolver.hpp"],
	deps = [":1d-heat-equation-finite-difference"],
)

cc_library(
	name = "1d-heat-equation-finite-difference",
	srcs = ["1DHeatEquationFiniteDifferenceSolver.cpp"],
	hdrs = ["1DHeatEquationFiniteDifferenceSolver.hpp"],
	visibility = ["//visibility:public"],
	deps = ["//src/finite-difference-arpack-version-2/arpackpp:arpackpp",
	        ":armadillo"],
)

cc_library(
	name = "2d-heat-equation-finite-difference",
	srcs = ["2DHeatEquationFiniteDifferenceSolver.cpp"],
	hdrs = ["2DHeatEquationFiniteDifferenceSolver.hpp"],
	visibility = ["//visibility:public"],
	deps = ["//src/igraph-0.7.1:igraph",
	        "//src/finite-difference-arpack-version-2/arpackpp:arpackpp",	
	        "//src/armadillo-7.600.2:armadillo", ":pde-data-types"],
	copts = ["-fopenmp",
		 "-Isrc/igraph-0.7.1/include",
		 "-Isrc/finite-difference-arpack-version-2",
		 "-Isrc/armadillo-7.600.2/usr/include"],
	linkopts = ["-fopenmp"],
)

cc_library(
	name = "arpack",
	srcs = ["libarpack.so"],
	copts = ["-Iarpackpp/external"],
)

cc_library(
	name = "pde-data-types",	
	srcs = ["PDEDataTypes.cpp"],
	hdrs = ["PDEDataTypes.hpp"],
	visibility = ["//visibility:public"],
	deps = ["//src/armadillo-7.600.2:armadillo"],
	copts = ["-O3",
		 "-Isrc/armadillo-7.600.2/usr/include"],
)
