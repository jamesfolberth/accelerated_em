
using Base.LinAlg.BLAS: ger!

#TODO: Verify that gradients are correct with Stephen's code
#TODO: gd_step!, compute_gradient for DiagCovMat

"""
Gradient descent driver
"""
function gd!{T,CM<:CovMat}(
      gmm::GMM{T,CM},
      X::Array{T,2};
      n_iter::Int=25,
      n_em_iter::Int=2,
      ll_tol::T=1e-3,
      print=false)
   
   n_dim, n_clust, n_ex = gmm_data_sanity(gmm, X)
   
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


function gd_step!{T,CM<:DiagCovMat}(
      gmm::GMM{T,CM},
      X::Array{T,2};
      var_thresh::T=1e-3,
      step_size::T=1e-4)
   
   n_dim, n_clust, n_ex = gmm_data_sanity(gmm, X)
   
   error("Not implemented!")
end
 
function gd_step!{T,CM<:FullCovMat}(
      gmm::GMM{T,CM},
      X::Array{T,2};
      chol_thresh::T=1e-1,
      step_size::T=1e-4)
  
   n_dim, n_clust, n_ex = gmm_data_sanity(gmm, X)

   ll, weight_grad, mean_grad, chol_grad = compute_gradient(gmm, X)
      
   # this is actually gradient ascent
   for k in 1:n_clust
      # this is a w_k update
      gmm.weights[k] = log(gmm.weights[k]) + step_size*weight_grad[k]
      gmm.means[k] += step_size*mean_grad[k]
   
      # TODO how to do this?
      R = gmm.covs[k].chol[:U] + step_size*chol_grad[k]
      gmm.covs[k].cov = R.'*R
      gmm.covs[k].chol = cholfact(gmm.covs[k].cov)
   end

   gmm.weights = exp(gmm.weights)
   gmm.weights /= sum(gmm.weights)

   return ll
end


function compute_gradient{T,CM<:DiagCovMat}(
      gmm::GMM{T,CM},
      X::Array{T,2})
   
   n_dim, n_clust, n_ex = gmm_data_sanity(gmm, X)
   
   error("Not implemented!")
end

function compute_gradient{T,CM<:FullCovMat}(
      gmm::GMM{T,CM},
      X::Array{T,2})

   n_dim, n_clust, n_ex = gmm_data_sanity(gmm, X)

   ## compute responsibilities ##
   cov_logdet = map(cm->logdet(cm.chol), gmm.covs) # logdet(Sigma)

   wrk = Array{T}(n_dim, n_ex) # used for storing R^{-T}(X-mean)
   xmmean = Array{Array{T,2},1}(n_clust) # storing (X-mean) for later use
   logpdf = Array{T}(n_ex, n_clust)  # log p(x_i | mean_k, Sigma_k)
   resp = Array{T}(n_ex, n_clust)    # responsibility of component j for example i
   
   # some initialization
   for k in 1:n_clust
      xmmean[k] = zeros(n_dim, n_ex)
   end
   
   # some heavy lifting
   for k in 1:n_clust
      broadcast!(-,xmmean[k],X.',gmm.means[k]) # (x-mean)
      wrk = gmm.covs[k].chol[:L] \ xmmean[k]   # R^{-T}*(x-mean)
      wrk .*= wrk                              # (x-mean)^T*prec*(x-mean)
      logpdf[:,k] = -T(0.5)*sum(wrk, 1) - T(0.5)*cov_logdet[k] -
         T(n_dim*0.5)*log(2*pi)
      resp[:,k] = gmm.weights[k]*exp(logpdf[:,k])
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
   
   weight_grad = Array{T}(n_clust)
   mean_grad = Array{Array{T,1},1}(n_clust)
   chol_grad = Array{Array{T,2},1}(n_clust) # gradient wrt Cholesky R
   outer_prod = Array{T}(zeros(n_dim, n_dim))
   for k in 1:n_clust # some initialization
      mean_grad[k] = Array{T}(zeros(n_dim))
      chol_grad[k] = Array{T}(zeros(n_dim, n_dim))
      chol_grad[k] -= rk[k]*inv(gmm.covs[k].chol[:L])
   end
   @inbounds for k in 1:n_clust
      weight_grad[k] = rk[k] - T(n_ex)*gmm.weights[k]
      for i in 1:n_ex
         xi = xmmean[k][:,i]
         mean_grad[k] += resp[i,k]*xi
         ger!(resp[i,k], xi, xi, outer_prod) # op += r_{ik}*xi*xi^T
      end
      mean_grad[k] = gmm.covs[k].chol \ mean_grad[k]
      # due to symmetry of outer_prod and Sigma, we can use "left division"
      outer_prod = (gmm.covs[k].chol \ outer_prod).' # (xi*xi')*inv(Sigma) 
      chol_grad[k] += gmm.covs[k].chol[:L] \ outer_prod
   end
   
   return ll, weight_grad, mean_grad, chol_grad
end



