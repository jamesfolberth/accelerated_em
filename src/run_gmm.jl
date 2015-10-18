#XXX
#srand(2718218)

using PyPlot

using EMAccel
using example_data

include("utils.jl")


## GMM development things ##
function run_2d()

   X, y = example_data.dist_2d_1(1000)
   clf()
   plot_data(X, y)

   #gmm = GMM(X; K=3, cov_type=:diag, mean_init_method=:kmeans)
   gmm = GMM(X; K=3, cov_type=:full, mean_init_method=:kmeans)

   println(typeof(gmm))

   em!(gmm, X, print=true)
   #gd!(gmm, X, print=true, n_em_iter=4)

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
   
   if n == 2
      clf()
      plot_data([X[:,1] X[:,2]], y)
   end 
   gmm = GMM(X; K=k, cov_type=:diag, mean_init_method=:kmeans)
   #gmm = GMM(X; K=k, cov_type=:full, mean_init_method=:kmeans)

   em!(gmm, X, print=true)
   #em!(gmm, X, print=true, ll_tol=-1.0, n_iter=100) # run forever...
   #gd!(gmm, X, print=true, n_em_iter=1)
   gmm.trained = true # force it, even if training fails
   
   println(gmm) 
   
   if n == 2
      plot_gmm_contours(gmm,
         [1.1*minimum(X[:,1]), 1.1*maximum(X[:,1]),
         1.1*minimum(X[:,2]), 1.1*maximum(X[:,2])])
   end

end

function run_compare_nd()

   n = 3
   k = 4
   N = 5000
   X, y = example_data.dist_nd_1(n, k, N, T=Float64, print=true)
   
   if n == 2
      figure(1)
      clf()
      plot_data([X[:,1] X[:,2]], y)
      figure(2)
      clf()
      plot_data([X[:,1] X[:,2]], y)
   end

   gmm1 = GMM(X; K=k, cov_type=:full, mean_init_method=:kmeans)
   #gmm2 = GMM(X; K=k, cov_type=:full, mean_init_method=:kmeans)
   gmm2 = deepcopy(gmm1)
   
   # force it!
   gmm1.trained = true
   gmm2.trained = true
   
   println("EM only")
   em!(gmm1, X, print=true, n_iter = 500)
   
   println("GD\n")
   gd!(gmm2, X, print=true, n_em_iter=2, n_iter=500)

   #println(gmm1)
   #println(gmm2)
   
   if n == 2
      figure(1)
      title("EM")
      plot_gmm_contours(gmm1,
         [1.1*minimum(X[:,1]), 1.1*maximum(X[:,1]),
          1.1*minimum(X[:,2]), 1.1*maximum(X[:,2])])

      figure(2)
      title("GD")
      plot_gmm_contours(gmm2,
         [1.1*minimum(X[:,1]), 1.1*maximum(X[:,1]),
          1.1*minimum(X[:,2]), 1.1*maximum(X[:,2])])
   end

end

#TODO make test dir?
function run_grad_check()
# {{{
   n = 2
   k = 4
   N = 500
   X, y = example_data.dist_nd_1(n, k, N, T=Float64, print=true)
   
   gmm = GMM(X; K=k, cov_type=:full, mean_init_method=:kmeans)
   
   for i in 1:0
      println(em_step!(gmm, X))
   end

   ## wk ##
   #function f(wk, gmm)
   #   gmm._wk[:] = wk[:]
   #   gmm.weights[:] = exp(wk)
   #   gmm.weights /= sum(gmm.weights)
   #   return compute_grad(gmm,X)[1]
   #end

   #function g(wk, gmm)
   #   gmm._wk[:] = wk[:]
   #   gmm.weights[:] = exp(wk)
   #   gmm.weights /= sum(gmm.weights)
   #   return compute_grad(gmm,X)[2]
   #end
   #grad_test.taylor_test(_x->f(_x,gmm), _x->g(_x,gmm),log(gmm.weights))
  

   ## means ##
   #ind = 1
   #function f(m, gmm)
   #   gmm.means[ind][:] = m[:]
   #   return compute_grad(gmm,X)[1]
   #end

   #function g(m, gmm)
   #   gmm.means[ind][:] = m[:]
   #   return compute_grad(gmm,X)[3][ind]
   #end
   #grad_test.taylor_test(_x->f(_x,gmm), _x->g(_x,gmm),gmm.means[ind])


   ## covs ##
   ind = 1
   # full
   function f(R, gmm)
      gmm.covs[ind].cov = R.'*R
      gmm.covs[ind].chol = cholfact(gmm.covs[ind].cov)
      return compute_grad(gmm,X)[1]
   end

   function g(R, gmm)
      gmm.covs[ind].cov = R.'*R
      gmm.covs[ind].chol = cholfact(gmm.covs[ind].cov)
      return compute_grad(gmm,X)[4][ind]
   end
   grad_test.taylor_test(_x->f(_x,gmm), _x->g(_x,gmm),full(gmm.covs[ind].chol[:U]))

# }}} 
end


# GMM
#run_2d()
#run_nd()
run_compare_nd()
#@time run_compare_nd()
#run_grad_check()

#end
