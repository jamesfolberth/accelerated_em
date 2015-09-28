"""
This module is an implementation of the EM algorithm for GMMs.  This is 
mainly for my understanding of the EM algorithm and GMMs.

There are other GMM modules in Julia:
   https://github.com/davidavdav/GaussianMixtures.jl
   https://github.com/lindahua/MixtureModels.jl
"""
#TODO visualize GMM somehow or some other metric for fit

module gmm

#XXX
#srand(2718218)

using PyPlot #using: brings names into current NS; import doesn't

# local files
include("types.jl")
include("em.jl")
include("utils.jl")

include("example_data.jl")


# development things
function run_2d()

   X, y = example_data.dist_2d_1(1000)
   clf()
   plot_data(X, y)

   #gmm = GMM(X; k=3, cov_type=:diag, mean_init_method=:kmeans)
   gmm = GMM(X; k=3, cov_type=:full, mean_init_method=:kmeans)

   em!(gmm, X, print=true)

   println(gmm)

   plot_gmm_contours(gmm,
      [1.1*minimum(X[:,1]), 1.1*maximum(X[:,1]),
       1.1*minimum(X[:,2]), 1.1*maximum(X[:,2])])

end

function run_nd()

   n = 5
   k = 4
   N = 5000
   X, y = example_data.dist_nd_1(n, k, N, T=Float64, print=true)
   
   #clf()
   #plot_data([X[:,1] X[:,2]], y)
   
   gmm = GMM(X; k=k, cov_type=:diag, mean_init_method=:kmeans)
   #gmm = GMM(X; k=k, cov_type=:full, mean_init_method=:kmeans)

   em!(gmm, X, print=true)

   println(gmm)
   
   #plot_gmm_contours(gmm,
   #   [1.1*minimum(X[:,1]), 1.1*maximum(X[:,1]),
   #    1.1*minimum(X[:,2]), 1.1*maximum(X[:,2])])

end

#run_2d()
run_nd()

end