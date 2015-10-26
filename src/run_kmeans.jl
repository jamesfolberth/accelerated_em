"""
This file is used for development.
"""

#XXX
#srand(2718218)
seed = rand(1:99999999)
#seed = 1
println("seed = $(seed)")
srand(seed)

using PyPlot 

#using EMAccel
reload("EMAccel")
import EMAccel
import MiscData

include("utils.jl")

## KMeans development things ##
# {{{
function run_kmeans_nd()

   n = 2
   k = 4
   N = 50000
   X, y = MiscData.RandCluster.dist_nd(n, k, N, T=Float64)
   m,s = standard_scaler!(X)
  
   if n == 2
      figure(1)
      clf()
      plot_data(X, y)
   end

   km = EMAccel.KMeans(X; K=k, mean_init_method=:kmpp)
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
      EMAccel.plot_means(km)
   end

   EMAccel.em!(km, X, print=true, n_iter=100)
   km.trained = true
   y_pred = EMAccel.soft_classify(km, X)
   dist = cluster_dist(km, X, y_pred)
   println("Sum of intra-cluster distances = $(sum(dist))")
   println()
   
   ##em!(km2, X, print=true, ll_tol=0.5)
   #EMAccel.gd!(km2, X, n_em_iter=0, print=true, n_iter=100)
   #km2.trained = true
   #y_pred2 = EMAccel.soft_classify(km2, X)
   #println()

   #EMAccel.em!(km3, X, print=true, ll_tol=0.5)
   EMAccel.nest2!(km3, X, n_em_iter=0, print=true, n_iter=100)
   km3.trained = true
   y_pred3 = EMAccel.soft_classify(km3, X)
   dist = cluster_dist(km, X, y_pred3)
   println("Sum of intra-cluster distances = $(sum(dist))")
 
   #EMAccel.em!(km, X, print=true)
   #km.trained = true
   #y_pred1 = EMAccel.soft_classify(km, X)
   #println()

   ##hard_em!(km, X)
   ##y_pred = hard_classify(km, X) 
 
   #EMAccel.em!(km2, X, print=true, ll_tol=0.5)
   ##EMAccel.gd!(km2, X, n_em_iter=2, print=true)
   #EMAccel.nest2!(km2, X, n_em_iter=0, print=true)
   ##EMAccel.nest2!(km, X, n_em_iter=2, print=true, ll_tol=1e-2)
   ##EMAccel.gd!(km, X, n_em_iter=0, print=true)
   #km2.trained = true
   #y_pred2 = EMAccel.soft_classify(km2, X)
   #println(km2)

   #println("\|y_pred - y_pred2\| = $(norm(y_pred-y_pred2))")

   if n == 2
      figure(2)
      clf()
      plot_data(X, y_pred)
      EMAccel.plot_means(km)
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
# {{{ Census1990
function run_census_kmeans()
   
   N = 50000
   k = 12
   X = MiscData.Census1990.read_array(nrows=N)
   m,s = standard_scaler!(X)
   println("Census1990 (subsample) data loaded.")
  
   km = EMAccel.KMeans(X; K=k, mean_init_method=:kmpp)
   km2 = deepcopy(km)
   km3 = deepcopy(km)
   
   EMAccel.em!(km, X, print=true)
   km.trained = true
   y_pred = EMAccel.soft_classify(km, X)
   dist = cluster_dist(km, X, y_pred)
   println("Sum of intra-cluster distances = $(sum(dist))")
 
   println()
   
   ##EMAccel.em!(km2, X, print=true, ll_tol=0.5)
   #EMAccel.gd!(km2, X, n_em_iter=0, print=true, n_iter=100)
   #km2.trained = true
   #y_pred2 = EMAccel.soft_classify(km2, X)
   #println()

   #EMAccel.em!(km3, X, print=true, ll_tol=0.5)
   EMAccel.nest2!(km3, X, n_em_iter=0, print=true, n_iter=100)
   km3.trained = true
   y_pred3 = EMAccel.soft_classify(km3, X)
   dist = cluster_dist(km3, X, y_pred3)
   println("Sum of intra-cluster distances = $(sum(dist))")
 
   return
end

# }}}

# {{{ Fisher's Iris
function run_iris_kmeans()
   
   X, y = MiscData.Iris.read_array()
   k = length(unique(y)) # k=3
   #k = 6
 
   m,s = standard_scaler!(X)
   
   km = EMAccel.KMeans(X; K=k, mean_init_method=:kmpp)
   km2 = deepcopy(km)
   km3 = deepcopy(km)
   
   EMAccel.em!(km, X, print=true)
   km.trained = true
   y_pred = EMAccel.soft_classify(km, X)
   dist = cluster_dist(km, X, y_pred)
   println("Sum of intra-cluster distances = $(sum(dist))")
   println()
   
   ##EMAccel.em!(km2, X, print=true, ll_tol=0.5)
   #EMAccel.gd!(km2, X, n_em_iter=0, print=true, n_iter=100)
   #km2.trained = true
   #y_pred2 = EMAccel.soft_classify(km2, X)
   #println()

   #EMAccel.em!(km3, X, print=true, ll_tol=0.5)
   EMAccel.nest2!(km3, X, n_em_iter=0, print=true, n_iter=100)
   km3.trained = true
   y_pred3 = EMAccel.soft_classify(km3, X)
   dist = cluster_dist(km3, X, y_pred3)
   println("Sum of intra-cluster distances = $(sum(dist))")
 
   return
end

# }}}

# {{{ LIBRAS
function run_libras_kmeans()
   
   X, y = MiscData.LIBRAS.read_array()
   k = length(unique(y)) # k=15
  
   m,s = standard_scaler!(X)
   
   km = EMAccel.KMeans(X; K=k, mean_init_method=:kmpp)
   km2 = deepcopy(km)
   km3 = deepcopy(km)
   
   EMAccel.em!(km, X, print=true, n_iter=100)
   km.trained = true
   y_pred = EMAccel.soft_classify(km, X)
   dist = cluster_dist(km, X, y_pred)
   println("Sum of intra-cluster distances = $(sum(dist))")
   println()
   
   ##EMAccel.em!(km2, X, print=true, ll_tol=0.5)
   #EMAccel.gd!(km2, X, n_em_iter=0, print=true, n_iter=100)
   #km2.trained = true
   #y_pred2 = EMAccel.soft_classify(km2, X)
   #println()

   #EMAccel.em!(km3, X, print=true, ll_tol=0.5)
   EMAccel.nest2!(km3, X, n_em_iter=0, print=true, n_iter=100)
   km3.trained = true
   y_pred3 = EMAccel.soft_classify(km3, X)
   dist = cluster_dist(km3, X, y_pred3)
   println("Sum of intra-cluster distances = $(sum(dist))")
 
   return
end

# }}}

# {{{ CMC
function run_cmc_kmeans()
   
   X, y = MiscData.CMC.read_array()
   k = length(unique(y)) # k=3
   
   m,s = standard_scaler!(X)

   km = EMAccel.KMeans(X; K=k, mean_init_method=:kmpp)
   km2 = deepcopy(km)
   km3 = deepcopy(km)
   
   EMAccel.em!(km, X, print=true)
   km.trained = true
   y_pred = EMAccel.soft_classify(km, X)
   dist = cluster_dist(km, X, y_pred)
   println("Sum of intra-cluster distances = $(sum(dist))")
   println()
   
   ##EMAccel.em!(km2, X, print=true, ll_tol=0.5)
   #EMAccel.gd!(km2, X, n_em_iter=0, print=true, n_iter=100)
   #km2.trained = true
   #y_pred2 = EMAccel.soft_classify(km2, X)
   #println()

   #EMAccel.em!(km3, X, print=true, ll_tol=0.5)
   EMAccel.nest2!(km3, X, n_em_iter=0, print=true, n_iter=100)
   km3.trained = true
   y_pred3 = EMAccel.soft_classify(km3, X)
   dist = cluster_dist(km3, X, y_pred3)
   println("Sum of intra-cluster distances = $(sum(dist))")
 
   return
end

# }}}

#TODO Wisconsin breast cancer dataset

# KMeans
#run_kmeans_nd()

# real data
#run_census_kmeans()
run_iris_kmeans()
#run_libras_kmeans()
#run_cmc_kmeans()

