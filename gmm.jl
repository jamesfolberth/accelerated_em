"""
This module has a few types of GMM and will eventually have a few different
algorithms for training them.  This is totally experimental!

There are other GMM modules in Julia:
   https://github.com/davidavdav/GaussianMixtures.jl
   https://github.com/lindahua/MixtureModels.jl
"""
#TODO visualize GMM somehow or some other metric for fit

module gmm

#XXX
srand(2718218)

using PyPlot #using: brings names into current NS; import doesn't

# local files
include("types.jl")
include("em.jl")
include("gd.jl")
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

   n = 2
   k = 4
   N = 5000
   X, y = example_data.dist_nd_1(n, k, N, T=Float64, print=true)
   
   clf()
   plot_data([X[:,1] X[:,2]], y)
   
   #gmm = GMM(X; k=k, cov_type=:diag, mean_init_method=:kmeans)
   gmm = GMM(X; k=k, cov_type=:full, mean_init_method=:kmeans)

   #em!(gmm, X, print=true)
   #em!(gmm, X, print=true, ll_tol=-1.0, n_iter=100) # run forever...
   gd!(gmm, X, print=true, n_em_iter=1)
   gmm.trained = true # force it, even if training fails
   
   println(gmm) 

   plot_gmm_contours(gmm,
      [1.1*minimum(X[:,1]), 1.1*maximum(X[:,1]),
       1.1*minimum(X[:,2]), 1.1*maximum(X[:,2])])

end

function run_compare_nd()

   n = 2
   k = 4
   N = 5000
   X, y = example_data.dist_nd_1(n, k, N, T=Float64, print=true)
   
   gmm1 = GMM(X; k=k, cov_type=:full, mean_init_method=:kmeans)
   gmm2 = GMM(X; k=k, cov_type=:full, mean_init_method=:kmeans)
   
   em!(gmm1, X, print=true)
   gd!(gmm2, X, print=true, n_em_iter=10, n_iter=50)

   println(gmm1)
   println(gmm2)

end



#run_2d()
#run_nd()
run_compare_nd()
#@time run_compare_nd()

end
