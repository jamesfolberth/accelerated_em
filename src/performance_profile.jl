
#TODO: we should make sure each method converges to the same value

#XXX
#srand(2718218)
seed = rand(1:99999999)
#seed = 1
println("seed = $(seed)")
srand(seed)

using PyPlot 

reload("EMAccel")
#import EMAccel
reload("MiscData")
#import MiscData

include("utils.jl")

"""
Create a (skeleton) performance profile.
"""
function perf_prof_skeleton{T<:Real}(
      perf_data::AbstractArray{T,2},
      tau::AbstractArray{T,1})
   
   np, ns = size(perf_data) # number of problems, number of solvers
   R = broadcast(/, perf_data, minimum(perf_data, 2))
   
   clf()
   rho = zeros(length(tau), ns)
   for s in 1:ns
      for (i,t) in enumerate(tau)
         rho[i,s] = 1/T(np)*T(sum(R[:,s] .<= t))
      end
      plot(tau, rho[:,s])  
   end
   xlabel("tau")
   ylabel("P(r[p,s] <= tau)")

   show()

   return
end


# Real data
# performance profile based on number of iterations (because it's easy)
function iter_perf_prof()
# {{{

   n_samples = 10
   MDlist = (MiscData.Iris, MiscData.LIBRAS, MiscData.CMC)
   solvers = (EMAccel.hard_em!, EMAccel.em!, EMAccel.gd!, EMAccel.nest2!)

   perf_data = zeros(length(MDlist), length(solvers))
   
   # do a bunch of solves
   for ns in 1:n_samples
      println("sample $(ns) of $(n_samples)")
      for (p,MDsub) in enumerate(MDlist)
         X,y = MDsub.read_array()
         k = length(unique(y))
  
         km = EMAccel.KMeans(X; K=k, mean_init_method=:kmpp)

         for (s,solver) in enumerate(solvers)
            km_copy = deepcopy(km)
            n_iter = solver(km_copy, X, n_iter=100)

            perf_data[p,s] += n_iter
         end
      end
   end 

   # build a performance profile plot
   perf_prof_skeleton(perf_data, collect(linspace(1,3)))
   axis([0.9, 3.1, -0.1, 1.1])
   
   legend(["hard_em!", "em!", "gd!", "nest2!"], loc="lower right")
   title("Performance profile: number of iterations")

   return
end
# }}}


# performance profile based on runtime
function time_perf_prof()
# {{{

   n_samples = 10
   MDlist = (MiscData.Iris, MiscData.LIBRAS, MiscData.CMC)
   solvers = (EMAccel.hard_em!, EMAccel.em!, EMAccel.gd!, EMAccel.nest2!)

   perf_data = zeros(length(MDlist), length(solvers))

   # force compilation of the solvers
   X,y = MiscData.Iris.read_array() # easy data
   k = length(unique(y))
   km = EMAccel.KMeans(X; K=k, mean_init_method=:kmpp)
   for (s,solver) in enumerate(solvers)
      km_copy = deepcopy(km)
      solver(km_copy, X)
   end

   # do a bunch of solves
   for ns in 1:n_samples
      println("sample $(ns) of $(n_samples)")
      for (p,MDsub) in enumerate(MDlist)
         X,y = MDsub.read_array()
         k = length(unique(y))
  
         km = EMAccel.KMeans(X; K=k, mean_init_method=:kmpp)

         for (s,solver) in enumerate(solvers)
            km_copy = deepcopy(km)
            t = @elapsed solver(km_copy, X, n_iter=100)
   
            perf_data[p,s] += t
         end
      end
   end 

   # build a performance profile plot
   perf_prof_skeleton(perf_data, collect(linspace(1,10)))
   axis([0.9, 10.1, -0.1, 1.1])
   
   legend(["hard_em!", "em!", "gd!", "nest2!"], loc="lower right")
   title("Performance profile: runtime")

   return
end
# }}}

# performance profile based on negative log-likelihood
function nll_perf_prof()
# {{{

   n_samples = 10
   MDlist = (MiscData.Iris, MiscData.LIBRAS, MiscData.CMC)
   solvers = (EMAccel.hard_em!, EMAccel.em!, EMAccel.gd!, EMAccel.nest2!)

   perf_data = zeros(length(MDlist), length(solvers))

   # do a bunch of solves
   for ns in 1:n_samples
      println("sample $(ns) of $(n_samples)")
      for (p,MDsub) in enumerate(MDlist)
         X,y = MDsub.read_array()
         k = length(unique(y))
  
         km = EMAccel.KMeans(X; K=k, mean_init_method=:kmpp)

         for (s,solver) in enumerate(solvers)
            km_copy = deepcopy(km)
            solver(km_copy, X, n_iter=100)
            ll = EMAccel.compute_ll(km_copy, X)

            perf_data[p,s] -= ll
         end
      end
   end 

   # build a performance profile plot
   perf_prof_skeleton(perf_data, collect(linspace(1,1.5)))
   axis([0.9, 1.6, -0.1, 1.1])
   
   legend(["hard_em!", "em!", "gd!", "nest2!"], loc="lower right")
   title("Performance profile: negative log-likelihood")

   return
end
# }}}

# performance profile based on sum of intra-cluster distances
function dist_perf_prof()
# {{{

   n_samples = 10
   MDlist = (MiscData.Iris, MiscData.LIBRAS, MiscData.CMC)
   solvers = (EMAccel.hard_em!, EMAccel.em!, EMAccel.gd!, EMAccel.nest2!)
   clusters = (EMAccel.hard_cluster, EMAccel.soft_cluster, EMAccel.soft_cluster, EMAccel.soft_cluster)

   perf_data = zeros(length(MDlist), length(solvers))

   # do a bunch of solves
   for ns in 1:n_samples
      println("sample $(ns) of $(n_samples)")
      for (p,MDsub) in enumerate(MDlist)
         X,y = MDsub.read_array()
         k = length(unique(y))
  
         km = EMAccel.KMeans(X; K=k, mean_init_method=:kmpp)

         for (s,solver) in enumerate(solvers)
            km_copy = deepcopy(km)
            solver(km_copy, X, n_iter=100)
            y_pred = clusters[s](km_copy, X)
            dist = cluster_dist(km_copy, X, y_pred)

            perf_data[p,s] += sum(dist)
         end
      end
   end 

   # build a performance profile plot
   perf_prof_skeleton(perf_data, collect(linspace(1,2.5)))
   axis([0.9, 2.6, -0.1, 1.1])
   
   legend(["hard_em!", "em!", "gd!", "nest2!"], loc="lower right")
   title("Performance profile: sum of intra-cluster distances")

   return
end
# }}}


# Manufactured data
# performance profile based on number of iterations (because it's easy)
function dist_nd_iter_perf_prof(dist_nd_args...; dist_nd_kwargs...)
# {{{

   n_samples = 30
   solvers = (EMAccel.hard_em!, EMAccel.em!, EMAccel.gd!, EMAccel.nest2!)

   perf_data = zeros(n_samples, length(solvers))
   
   # do a bunch of solves
   for ns in 1:n_samples
      println("sample $(ns) of $(n_samples)")
      X,y = MiscData.RandCluster.dist_nd(dist_nd_args...; dist_nd_kwargs...)
      k = length(unique(y))
  
      km = EMAccel.KMeans(X; K=k, mean_init_method=:kmpp)

      for (s,solver) in enumerate(solvers)
         km_copy = deepcopy(km)
         n_iter = solver(km_copy, X, n_iter=250)

         perf_data[ns,s] += n_iter
      end
   end 

   # build a performance profile plot
   perf_prof_skeleton(perf_data, collect(linspace(1,5)))
   axis([0.9, 5.1, -0.1, 1.1])
   
   legend(["hard_em!", "em!", "gd!", "nest2!"], loc="lower right")
   title("Performance profile: number of iterations")

   return
end
# }}}

# performance profile based on runtime
function dist_nd_time_perf_prof(dist_nd_args...; dist_nd_kwargs...)
# {{{

   n_samples = 30
   solvers = (EMAccel.hard_em!, EMAccel.em!, EMAccel.gd!, EMAccel.nest2!)

   perf_data = zeros(n_samples, length(solvers))

   X,y = MiscData.RandCluster.dist_nd(dist_nd_args...; dist_nd_kwargs...)
   k = length(unique(y))
   km = EMAccel.KMeans(X; K=k, mean_init_method=:kmpp)
   for (s,solver) in enumerate(solvers)
      km_copy = deepcopy(km)
      solver(km_copy, X)
   end

   # do a bunch of solves
   for ns in 1:n_samples
      println("sample $(ns) of $(n_samples)")
      X,y = MiscData.RandCluster.dist_nd(dist_nd_args...; dist_nd_kwargs...)
      k = length(unique(y))
  
      km = EMAccel.KMeans(X; K=k, mean_init_method=:kmpp)

      for (s,solver) in enumerate(solvers)
         km_copy = deepcopy(km)
         t = @elapsed solver(km_copy, X, n_iter=250)

         perf_data[ns,s] += t
      end
   end 

   # build a performance profile plot
   perf_prof_skeleton(perf_data, collect(linspace(1,10)))
   axis([0.9, 10.1, -0.1, 1.1])
   
   legend(["hard_em!", "em!", "gd!", "nest2!"], loc="lower right")
   title("Performance profile: runtime")

   return
end
# }}}

# performance profile based on negative log-likelihood
function dist_nd_nll_perf_prof(dist_nd_args...; dist_nd_kwargs...)
# {{{

   n_samples = 30
   solvers = (EMAccel.hard_em!, EMAccel.em!, EMAccel.gd!, EMAccel.nest2!)

   perf_data = zeros(n_samples, length(solvers))
   
   # do a bunch of solves
   for ns in 1:n_samples
      println("sample $(ns) of $(n_samples)")
      X,y = MiscData.RandCluster.dist_nd(dist_nd_args...; dist_nd_kwargs...)
      k = length(unique(y))
  
      km = EMAccel.KMeans(X; K=k, mean_init_method=:kmpp)

      for (s,solver) in enumerate(solvers)
         km_copy = deepcopy(km)
         solver(km_copy, X, n_iter=250)
         ll = EMAccel.compute_ll(km_copy, X)

         perf_data[ns,s] -= ll
      end
   end 

   # build a performance profile plot
   perf_prof_skeleton(perf_data, collect(linspace(1,1.5)))
   axis([0.9, 1.6, -0.1, 1.1])
   
   legend(["hard_em!", "em!", "gd!", "nest2!"], loc="lower right")
   title("Performance profile: negative log-likelihood")

   return
end
# }}}



# real data
#iter_perf_prof()
#time_perf_prof()
#nll_perf_prof()
#dist_perf_prof()


# made up data
#f = dist_nd_iter_perf_prof
f = dist_nd_time_perf_prof
#f = dist_nd_nll_perf_prof

# nest2! wins (by a bit) in iter and time for this severely unbalanced dataset.
#f(10,2,1000,cov_type=:wishart,mean_std=10,noise_std=5,weights=[.99 .01])
#f(30,2,5000,cov_type=:wishart,mean_std=10,noise_std=5,weights=[.99 .01])

# nest2! wins (by a bit) in iter and time for this severely unbalanced dataset.
# "n=30_K=4_N=3000_unbalanced.pdf" nest2 wins bit a bit more
# "n=30_K=4_N=3000_unbalanced_iter.pdf" nest2 wins bit a bit more
#f(30,4,3000,cov_type=:spherical,mean_std=10,noise_std=5,weights=[.7 .1 .1 .1])
#f(3,4,300,cov_type=:spherical,mean_std=10,noise_std=5,weights=[.7 .1 .1 .1])

# "n=100_K=4_N=10000_unbalanced.pdf" nest2 wins bit a bit more
#f(100,4,10000,cov_type=:wishart,mean_std=10,noise_std=5,weights=[.97 .01 .01 .01])

# "n=3_K=40_N=3000_iter.pdf" nest2 wins by a bit
f(3,40,3000,cov_type=:spherical,mean_std=10,noise_std=5)

