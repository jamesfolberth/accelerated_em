"""
This module is an implementation of the EM algorithm for GMMs.  This is 
mainly for my understanding of the EM algorithm and GMMs.
"""
#TODO visualize GMM somehow

module gmm_em

#XXX
srand(2718218)

using PyPlot #using: brings names into current NS; import doesn't
import Clustering # kmeans to initialize

# local files
include("types.jl")
include("em.jl")
include("utils.jl")

include("example_data.jl")


# development things
function run_2d()

   X, y = example_data.dist_2d_1(1000)
   clf()
   #plot_data(X)
   plot_data(X, y)

   #gmm = GMM(X; k=3, cov_type=:diag, mean_init_method=:zeros)
   gmm = GMM(X; k=3, cov_type=:diag, mean_init_method=:kmeans)
   #gmm = GMM(X; k=3, cov_type=:diag, mean_init_method=:rand)

   em!(gmm, X)

   println(gmm)

   plot_gmm_contours(gmm, [-10 10 -10 10])

end

run_2d()

end
