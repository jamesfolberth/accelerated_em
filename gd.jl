
using Base.LinAlg.BLAS: ger!

"""
Gradient descent driver
"""
function gd!{T<:AbstractFloat}(
      gmm::GMM{T},
      X::Array{T,2};
      n_iter::Int=25,
      n_em_iter::Int=2,
      ll_tol::T=1e-3,
      print=false)

   n_ex, n_dim = size(X)
   k = size(gmm.means, 1) # num components
   
   # initially do a few EM steps
   prev_ll = -T(Inf)
   for it in 1:n_em_iter
      prev_ll = em_step!(gmm, X)
      println("(EM step) log-likelihood = $(prev_ll)")
   end

   ll_diff = T(0.0)
   for it in 1:n_iter
      #TODO line search
      ll = gd_step!(gmm, X, step_size=1e-4/(1+it)^(.7))

      if print
         println("log-likelihood = $(ll)")
         #TODO print convergence rates
      end
     
      # check log-likelihood convergence
      ll_diff = abs(prev_ll - ll)
      #if ll_diff < ll_tol
      #   break
      #end
      prev_ll = ll
   end
   
   if ll_diff > ll_tol
      warn("Log-likelihood has not reached convergence criterion of $(ll_tol) in $(n_iter) iterations.  GD may not have converged!")

   else
      gmm.trained = true
   end

   return gmm
end


#TODO: I'd like to cleanly break this function into specialized versions
#      for diag and full matrices.  GaussianMixtures.jl has two type
#      parameters to do this.  Is this the Julian way to go?
function gd_step!{T<:AbstractFloat}(
      gmm::GMM{T},
      X::Array{T,2};
      var_thresh::T=1e-3,
      chol_thresh::T=1e-1,
      step_size::T=1e-4)
  
   k = size(gmm.means, 1) # num components

   if gmm.cov_type == :diag
      error("Not implemented!")

   elseif gmm.cov_type == :full
      ll, weight_grad, mean_grad, chol_grad = compute_gradient(gmm, X)
      
      # this is actually gradient ascent
      for j in 1:k
         # this is a w_k update
         gmm.weights[j] = log(gmm.weights[j]) + step_size*weight_grad[j]
         gmm.means[j] += step_size*mean_grad[j]
   
         # TODO how to do this?
         R = gmm.covs[j].chol[:U] + step_size*chol_grad[j]
         gmm.covs[j].cov = R.'*R
         gmm.covs[j].chol = cholfact(gmm.covs[j].cov)
      end

     gmm.weights = exp(gmm.weights)
     gmm.weights /= sum(gmm.weights)

   else
      error("Unknown covariance type $(gmm.cov_type).")
   end
   
   return ll
end


function compute_gradient{T<:AbstractFloat}(
      gmm::GMM{T},
      X::Array{T,2})

   n_ex, n_dim = size(X)
   k = size(gmm.means, 1) # num components

   if gmm.cov_type == :diag
      error("Not implemented!")

   elseif gmm.cov_type == :full
      ## compute responsibilities ##
      cov_logdet = map(cm->logdet(cm.chol), gmm.covs) # logdet(Sigma)

      wrk = Array{T}(n_dim, n_ex) # used for storing R^{-T}(X-mean)
      xmmean = Array{Array{T,2},1}(k) # storing (X-mean) for later use
      logpdf = Array{T}(n_ex, k)  # log p(x_i | mean_k, Sigma_k)
      resp = Array{T}(n_ex, k)    # responsibility of component j for example i
      
      # some initialization
      for j in 1:k
         xmmean[j] = zeros(n_dim, n_ex)
      end
      
      # some heavy lifting
      for j in 1:k
         broadcast!(-,xmmean[j],X.',gmm.means[j]) # (x-mean)
         wrk = gmm.covs[j].chol[:L] \ xmmean[j]   # R^{-T}*(x-mean)
         wrk .*= wrk                              # (x-mean)^T*prec*(x-mean)
         logpdf[:,j] = -T(0.5)*sum(wrk, 1) - T(0.5)*cov_logdet[j] -
            T(n_dim*0.5)*log(2*pi)
         resp[:,j] = gmm.weights[j]*exp(logpdf[:,j])
      end
      
      # log-likelihood before gradient update
      # note that resp isn't yet normalized
      ll = reshape(sum(log(sum(resp, 2)), 1)/T(n_ex), 1)[1]

      # Baye's rule to normalize responsibilities
      broadcast!(/, resp, resp, sum(resp, 2))
      
      # sometimes zero-sum responsibilities are introduced
      resp[find(isnan(resp))] = T(0.0) # set NaNs to zero
      

      ## compute terms of gradient ##
      rk = sum(resp, 1) # \sum_{i=1}^N r_{ik}
      
      weight_grad = Array{T}(k)
      mean_grad = Array{Array{T,1},1}(k)
      chol_grad = Array{Array{T,2},1}(k) # gradient wrt Cholesky R
      outer_prod = Array{T}(zeros(n_dim, n_dim))
      for j in 1:k # some initialization
         mean_grad[j] = Array{T}(zeros(n_dim))
         chol_grad[j] = Array{T}(zeros(n_dim, n_dim))
         chol_grad[j] -= rk[j]*inv(gmm.covs[j].chol[:L])
      end
      @inbounds for j in 1:k
         weight_grad[j] = rk[j] - T(n_ex)*gmm.weights[j]
         for i in 1:n_ex
            xi = xmmean[j][:,i]
            mean_grad[j] += resp[i,j]*xi
            ger!(resp[i,j], xi, xi, outer_prod) # op += r_{ij}*xi*xi^T
         end
         mean_grad[j] = gmm.covs[j].chol \ mean_grad[j]
         # due to symmetry of outer_prod and Sigma, we can use "left division"
         outer_prod = (gmm.covs[j].chol \ outer_prod).' # (xi*xi')*inv(Sigma) 
         chol_grad[j] += gmm.covs[j].chol[:L] \ outer_prod
      end
      
      return ll, weight_grad, mean_grad, chol_grad

   else
      error("Unknown covariance type $(gmm.cov_type).")
   end

end



