using Base.LinAlg.BLAS: ger!

#TODO: gd_step!, compute_grad for DiagCovMat

## GMM - simple GD ##
# {{{
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
   
   n_dim, n_clust, n_ex = data_sanity(gmm, X)
   
   # initially do a few EM steps
   prev_ll = -T(Inf)
   for it in 1:n_em_iter
      prev_ll = em_step!(gmm, X)
      println("em!: log-likelihood = $(prev_ll)")
   end

   ll_diff = T(0)
   bt_step = T(1)
   for it in 1:n_iter
      # naive step size
      #ll = gd_step!(gmm, X, step_size=1e-4/(1+it)^(.7))
      
      # backtracking line search
      if it == 1
         ll, gmm, bt_step = bt_ls_step(gmm, X)
      else
         ll, gmm, bt_step = bt_ls_step(gmm, X, alpha=bt_step)
         # hacky way to attempt to grow step size
         #ll, gmm, bt_step = bt_ls_step(gmm, X, alpha=2.0*bt_step)
      end

      if print
         println("gd!: log-likelihood = $(ll)")
         #TODO print convergence rates
      end
     
      # check log-likelihood convergence
      ll_diff = abs(prev_ll - ll)
      if ll_diff < ll_tol
         break
      end
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
   
   n_dim, n_clust, n_ex = data_sanity(gmm, X)
   
   error("Not implemented!")
end
 
function gd_step!{T,CM<:FullCovMat}(
      gmm::GMM{T,CM},
      X::Array{T,2};
      chol_thresh::T=1e-1,
      step_size::T=1e-4)
  
   n_dim, n_clust, n_ex = data_sanity(gmm, X)
   
   gmm._wk[:] = log(gmm.weights[:])
   ll, wk_grad, mean_grad, chol_grad = compute_grad(gmm, X)
      
   # this is actually gradient ascent
   for k in 1:n_clust
      gmm._wk[k] += step_size*wk_grad[k]

      gmm.means[k] += step_size*mean_grad[k]
   
      # TODO how to do this?
      R = gmm.covs[k].chol[:U] + step_size*chol_grad[k]
      gmm.covs[k].cov = R.'*R
      gmm.covs[k].chol = cholfact(gmm.covs[k].cov)
   end

   gmm.weights[:] = exp(gmm._wk[:])
   gmm.weights /= sum(gmm.weights)

   return ll
end


## line search ##
# for steepest descent
function bt_ls_step{T,CM<:FullCovMat}(
      gmm::GMM{T,CM},
      X::Array{T,2};
      alpha::T=1e-2,
      rho::T=0.5,
      c::T=1e-4)
   
   function step_gmm!(gmm, wk_grad, mean_grad, chol_grad, alpha)
      n_dim, n_clust, n_ex = data_sanity(gmm, X)

      gmm._wk[:] = log(gmm.weights[:])
      for k in 1:n_clust
         gmm._wk[k] += alpha*wk_grad[k]

         gmm.means[k] += alpha*mean_grad[k]
      
         # TODO how to do this?
         R = gmm.covs[k].chol[:U] + alpha*chol_grad[k]
         gmm.covs[k].cov = R.'*R
         gmm.covs[k].chol = cholfact(gmm.covs[k].cov)
      end

      gmm.weights[:] = exp(gmm._wk[:])
      gmm.weights /= sum(gmm.weights)
   end
   
   n_dim, n_clust, n_ex = data_sanity(gmm, X)
   ll, wk_grad, mean_grad, chol_grad = compute_grad(gmm, X)
   
   alpha_k = alpha
   _gmm = deepcopy(gmm)
   step_gmm!(_gmm, wk_grad, mean_grad, chol_grad, alpha_k)
   _ll = compute_ll(_gmm, X)

   grad_ip = T(0)
   for k in 1:n_clust
      grad_ip += wk_grad[k]^2
      grad_ip += sum(mean_grad[k].^2)
      grad_ip += sum(chol_grad[k].^2)
   end
   #println("  grad_ip = $(grad_ip)")

   while isnan(_ll) || _ll < ll + c*alpha_k*grad_ip
      alpha_k *= rho 

      _gmm = deepcopy(gmm)
      step_gmm!(_gmm, wk_grad, mean_grad, chol_grad, alpha_k)
      _ll = compute_ll(_gmm, X)
      #println("  alpha_k = $(alpha_k), _ll = $(_ll), diff = $(_ll-ll-c*alpha_k*grad_ip)")
      
      if alpha_k < eps(ll)
         return ll, gmm, alpha_k # sufficient decrease not found
      end
   end
   
   #gmm = deepcopy(_gmm) #TODO is there a better way?
   return _ll, _gmm, alpha_k
end

# }}}

## Gradients ##
# {{{
function compute_grad{T,CM<:DiagCovMat}(
      gmm::GMM{T,CM},
      X::Array{T,2})
   
   n_dim, n_clust, n_ex = data_sanity(gmm, X)
   
   error("Not implemented!")
end

function compute_grad{T,CM<:FullCovMat}(
      gmm::GMM{T,CM},
      X::Array{T,2})

   n_dim, n_clust, n_ex = data_sanity(gmm, X)

   ## compute responsibilities ##
   cov_logdet = map(cm->logdet(cm.chol), gmm.covs) # logdet(Sigma)

   wrk = Array{T}(n_dim, n_ex) # used for storing R^{-T}(X-mean)
   xmmean = Array{Array{T,2},1}(n_clust) # storing (X-mean) for later use
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
      resp[:,k] = -T(0.5)*sum(wrk, 1) - T(0.5)*cov_logdet[k] -
         T(n_dim*0.5)*log(2*pi)
   end
   
   # log-likelihood before gradient update
   # log-sum-exp trick
   m = maximum(resp,2)
   broadcast!(-, resp, resp, m) # don't have to unshift, since we normalize later
   for k in 1:n_clust
      resp[:,k] = gmm.weights[k]*exp(resp[:,k]) 
   end
   ll = (sum(m) + sum(log(sum(resp,2))))/T(n_ex)
 
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
      outer_prod *= T(0)
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

# }}}

