#XXX
#srand(2718218)

using PyPlot

#using EMAccel
reload("EMAccel")
import EMAccel
import MiscData.RandCluster

include("utils.jl")


## GMM development things ##
function run_nd()

   n = 2
   k = 4
   N = 5000
   X, y = RandCluster.dist_nd(n, k, N, T=Float64, print=true)
   
   if n == 2
      clf()
      plot_data([X[:,1] X[:,2]], y)
   end 

   gmm = EMAccel.GMM(X; K=k, cov_type=:diag, mean_init_method=:kmeans)
   #gmm = EMAccel.GMM(X; K=k, cov_type=:full, mean_init_method=:kmeans)

   EMAccel.em!(gmm, X, print=true)
   #EMAccel.em!(gmm, X, print=true, ll_tol=-1.0, n_iter=100) # run forever...
   #EMAccel.gd!(gmm, X, print=true, n_em_iter=1)
   gmm.trained = true # force it, even if training fails
   
   println(gmm) 
   
   if n == 2
      EMAccel.plot_gmm_contours(gmm,
         [1.1*minimum(X[:,1]), 1.1*maximum(X[:,1]),
         1.1*minimum(X[:,2]), 1.1*maximum(X[:,2])])
   end

end

function run_compare_nd()

   n = 3
   k = 4
   N = 5000
   X, y = RandCluster.dist_nd(n, k, N, T=Float64, print=true)
   
   if n == 2
      figure(1)
      clf()
      plot_data([X[:,1] X[:,2]], y)
      figure(2)
      clf()
      plot_data([X[:,1] X[:,2]], y)
   end

   gmm1 = EMAccel.GMM(X; K=k, cov_type=:full, mean_init_method=:kmeans)
   #gmm2 = EMAccel.GMM(X; K=k, cov_type=:full, mean_init_method=:kmeans)
   gmm2 = deepcopy(gmm1)
   
   # force it!
   gmm1.trained = true
   gmm2.trained = true
   
   println("EM only")
   EMAccel.em!(gmm1, X, print=true, n_iter = 500)
   
   println("GD\n")
   EMAccel.gd!(gmm2, X, print=true, n_em_iter=2, n_iter=500)

   #println(gmm1)
   #println(gmm2)
   
   if n == 2
      figure(1)
      title("EM")
      EMAccel.plot_gmm_contours(gmm1,
         [1.1*minimum(X[:,1]), 1.1*maximum(X[:,1]),
          1.1*minimum(X[:,2]), 1.1*maximum(X[:,2])])

      figure(2)
      title("GD")
      EMAccel.plot_gmm_contours(gmm2,
         [1.1*minimum(X[:,1]), 1.1*maximum(X[:,1]),
          1.1*minimum(X[:,2]), 1.1*maximum(X[:,2])])
   end

end

# GMM
#run_nd()
run_compare_nd()
#@time run_compare_nd()

#end
