"""
Routines to generate some simple example data.
"""

module example_data

import Distributions


"""
Draw N samples from a fixed 2d mixture

Args:
   N (Integer) - number of samples to draw

Return:
   X () - (2,N) array of samples
"""
function dist_2d_1(N::Integer)

   K = 3
   mix = [.25; .25; .5]
   mean = [0 0; 1 5; -5 -5].'

   cov1 = [.5 0.; 0. .1]
   #cov1 = [.05 0.; 0. .0000001]
   cov2 = [1. 1.; 1. 2.]
   cov3 = [1. 0.; 0. 1.]
   covs = Array[cov1, cov2, cov3]

   # precompute spectral decomposition
   # precompute linear part of affine transformation mapping N(0,1) 
   # to N(mu, Sigma): x <- sqrt(D)*U*x+mu
   Q = Array{Any}(size(covs,1))
   for ind in 1:size(covs,1)
      cov = covs[ind]
      eig = eigfact(cov)
      Q[ind] = eig[:vectors]*diagm(sqrt(eig[:values]))
   end

   T = typeof(1.)
   X = randn((2,N))
   y = Array{Int}(N)
   
   mix_cumsum = cumsum(mix)
   for n in 1:N
      # select a cluster from which to sample
      u = rand()
      k = searchsortedfirst(mix_cumsum, u)
      
      # sample from that cluster
      X[:,n]  = Q[k]*X[:,n] + mean[:,k]
      y[n] = k

   end

   return X.', y
end


"""
Sample N points from a random mixture in n dimensions with k components.
"""
function dist_nd_1(n::Integer, K::Integer, N::Integer;
      T=Float64, cov_type=:wishart, print=false)
   
   # mixture weights, sum to one via softmax
   mix = Array{T}(randn((K,)))
   mix = exp(mix)
   mix /= sum(mix)

   # means
   mean = Array{T}(10*randn((n, K)))

   # covariances
   covs = Array{Array{T,2},1}(K)

   if cov_type == :simple
      # fill with A \gets A + A^T + nI
      for k in 1:K
         A = Array{T}(rand(n,n))
         A += A.'
         A += T(n)*Array{T}(eye(n))
         covs[k] = A
      end

   elseif cov_type == :wishart
      # or fill with samples from Wishart_n(nI, n)
      S = T(n)*Array{T}(eye(n))
      W_n = Distributions.Wishart(T(n), S)
      for k in 1:K
         covs[k] = rand(W_n)
      end
   else
      error("unknown covariance type $(cov_type).")
   end

   # precompute spectral decomposition
   # precompute linear part of affine transformation mapping N(0,1) 
   # to N(mu, Sigma): x <- sqrt(D)*U*x+mu
   Q = Array{Any}(K)
   for ind in 1:K
      cov = covs[ind]
      eig = eigfact(cov)
      Q[ind] = eig[:vectors]*diagm(sqrt(eig[:values]))
   end

   X = randn((n,N))
   y = Array{Int}(N)
   
   mix_cumsum = cumsum(mix)
   for ind in 1:N
      # select a cluster from which to sample
      u = rand()
      k = searchsortedfirst(mix_cumsum, u)
      
      # sample from that cluster
      X[:,ind]  = Q[k]*X[:,ind] + mean[:,k]
      y[ind] = k

   end

   if print
      println("Example data in $(n) dimensions with $(K) components using cov_type=$(cov_type).")
      for k in 1:K
         println("  Component $(k):")
         println("    weight:     $(mix[k])")

         println("    mean:")
         pretty_print_vector(6, mean[:,k])

         println("    cov (full):")
         pretty_print_matrix(6, covs[k])

      end
   end

   return X.', y
end

#TODO merge this with gmm.pretty_print_*
function pretty_print_vector(indent_level, x)
   for val in x
      println(join([repeat(" ",indent_level), @sprintf "% 7.3f" val]))
   end
end

function pretty_print_matrix(indent_level, mat)
   for i in 1:size(mat,1)
      print(repeat(" ", indent_level))
      for val in mat[i,:]
         print(@sprintf "% 7.3f  " val)
      end
      println("")
   end
end

end
