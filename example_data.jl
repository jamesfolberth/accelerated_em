"""
Routines to generate some simple example data.
"""

module example_data

"""
Draw N samples from a fixed 2d mixture

Args:
   N (Int) - number of samples to draw

Return:
   X () - (2,N) array of samples
"""
function dist_2d_1(N::Int)

   K = 3
   mix = [.25; .25; .5]
   mean = [0 0; 1 5; -5 -5].'

   cov1 = [.5 0.; 0. .1]
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




end
