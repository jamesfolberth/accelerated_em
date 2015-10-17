"""
This module has an implementation of k-means with a few different training methods.
This is totally experimental!

There are other k-means implementations in Julia:
https://github.com/JuliaStats/Clustering.jl
"""

module kmeans

#XXX
#srand(2718218)
seed = rand(1:99999999)
#seed = 1
println("seed = $(seed)")
srand(seed)

# these are just for plotting
using PyCall
using PyPlot #using: brings names into current NS; import doesn't
@pyimport matplotlib.patches as mpatches

# some real data
import MiscData.Census1990


# local files
include("kmeans/types.jl")
include("kmeans/em.jl")
include("kmeans/gd.jl")
include("kmeans/utils.jl")

include("utils.jl")
include("example_data.jl")


## KMeans development things ##
# {{{
function run_kmeans_nd()

   n = 3
   k = 8
   N = 50000
   X, y = example_data.dist_nd_1(n, k, N, T=Float64)
  
   if n == 2
      figure(1)
      clf()
      plot_data(X, y)
   end

   km = KMeans(X; K=k, mean_init_method=:kmpp)
   km2 = deepcopy(km)
   km3 = deepcopy(km)
   
   # steal means from Clustering's kmeans
   # note that they use different stopping criteria for EM
   #kmr = kmeans(X.', k, init=:kmpp, maxiter=0) # to check kmeans++; seems good
   #for i in 1:k
   #   km.means[i] = vec(kmr.centers[:,i])
   #   km2.means[i] = vec(kmr.centers[:,i])
   #end
   
   if n == 2
      plot_means(km)
   end

   em!(km, X, print=true, n_iter=100)
   km.trained = true
   y_pred = soft_classify(km, X)
   println()
   
   ##em!(km2, X, print=true, ll_tol=0.5)
   #gd!(km2, X, n_em_iter=0, print=true, n_iter=100)
   #km2.trained = true
   #y_pred2 = soft_classify(km2, X)
   #println()

   #em!(km3, X, print=true, ll_tol=0.5)
   nest2!(km3, X, n_em_iter=0, print=true, n_iter=100)
   km3.trained = true
   y_pred3 = soft_classify(km3, X)

   #em!(km, X, print=true)
   #km.trained = true
   #y_pred1 = soft_classify(km, X)
   #println()

   ##hard_em!(km, X)
   ##y_pred = hard_classify(km, X) 
 
   #em!(km2, X, print=true, ll_tol=0.5)
   ##gd!(km2, X, n_em_iter=2, print=true)
   #nest2!(km2, X, n_em_iter=0, print=true)
   ##nest2!(km, X, n_em_iter=2, print=true, ll_tol=1e-2)
   ##gd!(km, X, n_em_iter=0, print=true)
   #km2.trained = true
   #y_pred2 = soft_classify(km2, X)
   #println(km2)

   #println("\|y_pred - y_pred2\| = $(norm(y_pred-y_pred2))")

   if n == 2
      figure(2)
      clf()
      plot_data(X, y_pred)
      plot_means(km)
   end

   #if n == 2
   #   figure(3)
   #   clf()
   #   title("Clustering.jl:kmeans")
   #   kmr = kmeans(X.', k, init=:kmpp)
   #   plot_data(X, kmr.assignments)
   #end
   
end


# }}}

## Real data ##
# {{{
function run_census_kmeans()
   
   N = 5000
   k = 12
   X = Census1990.read_array(nrows=N)
   println("Census1990 (subsample) data loaded.")
  
   km = KMeans(X; K=k, mean_init_method=:kmpp)
   km2 = deepcopy(km)
   km3 = deepcopy(km)
   
   em!(km, X, print=true)
   km.trained = true
   y_pred = soft_classify(km, X)
   println()
   
   ##em!(km2, X, print=true, ll_tol=0.5)
   #gd!(km2, X, n_em_iter=0, print=true, n_iter=100)
   #km2.trained = true
   #y_pred2 = soft_classify(km2, X)
   #println()

   #em!(km3, X, print=true, ll_tol=0.5)
   nest2!(km3, X, n_em_iter=0, print=true, n_iter=100)
   km3.trained = true
   y_pred3 = soft_classify(km3, X)

end

# }}}

# KMeans
#run_kmeans_nd()

# real data
run_census_kmeans()

end
