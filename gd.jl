
using Base.LinAlg.BLAS: ger!

#TODO: Verify that gradients are correct with Stephen's code
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


## Gradients ##
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


## KMeans - simple GD ##
# {{{
"""
Gradient descent driver
"""
function gd!{T}(
      km::KMeans{T},
      X::Array{T,2};
      n_iter::Int=25,
      n_em_iter::Int=2,
      ll_tol::T=1e-3,
      print=false)
   
   n_dim, n_clust, n_ex = data_sanity(km, X)
   
   # initially do a few EM steps
   prev_ll = -T(Inf)
   for it in 1:n_em_iter
      prev_ll = em_step!(km, X)
      println("em!: log-likelihood = $(prev_ll)")
   end

   ll_diff = T(0)
   bt_step = T(1)
   for it in 1:n_iter
      # naive step size
      #ll = gd_step!(km, X, step_size=:em_step)
      #ll = gd_step!(km, X, step_size=1e-4/(1+it)^(.7))
      
      # backtracking line search
      #TODO
      if it == 1
         ll, bt_step = bt_ls_step!(km, X, alpha=bt_step)
      else
         ll, bt_step = bt_ls_step!(km, X, alpha=bt_step)
         # hacky way to attempt to grow step size
         #ll, bt_step = bt_ls_step(km, X, alpha=2.0*bt_step)
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
      km.trained = true
   end

   return km
end


"""
gradient step for soft k-means
"""
function gd_step!{T}(
      km::KMeans{T},
      X::Array{T,2};
      step_size=1e-4)

   n_dim, n_clust, n_ex = data_sanity(km, X)
   
   sigma = T(1)
   
   ll, mean_grad, resp = compute_grad(km, X)
   rk = sum(resp,1)

   if step_size == :em_step
      for k in 1:n_clust
         eta = sigma^2/rk[k] # this step size recovers EM
         km.means[k] += eta*mean_grad[k]
      end

   elseif step_size <: T
      for k in 1:n_clust
         km.means[k] += step_size*mean_grad[k]
      end

   else
      error("Bad step size $(step_size)")
   end

   return ll

end

"""
Backtracking line search for KMeans GD
"""
function bt_ls_step!{T}(
      km::KMeans{T},
      X::Array{T,2};
      alpha::T=1e-2,
      rho::T=0.5,
      c::T=1e-4)
   
   function step_km!(km, mean_grad, alpha)
      n_dim, n_clust, n_ex = data_sanity(km, X)

      for k in 1:n_clust
         km.means[k] += alpha*mean_grad[k]
      end
   end
   
   n_dim, n_clust, n_ex = data_sanity(km, X)
   ll, mean_grad, resp  = compute_grad(km, X)
   
   alpha_k = alpha
   _km = deepcopy(km)
   step_km!(_km, mean_grad, alpha_k)
   _ll = compute_ll(_km, X)

   grad_ip = T(0)
   for k in 1:n_clust
      grad_ip += sumabs2(mean_grad[k])
   end
   #println("  grad_ip = $(grad_ip)")

   while isnan(_ll) || _ll < ll + c*alpha_k*grad_ip
      alpha_k *= rho 

      _km = deepcopy(km)
      step_km!(_km, mean_grad, alpha_k)
      _ll = compute_ll(_km, X)
      #println("  alpha_k = $(alpha_k), _ll = $(_ll), diff = $(_ll-ll-c*alpha_k*grad_ip)")
      
      if alpha_k < eps(ll)
         return ll, alpha_k # sufficient decrease not found
      end
   end
   
   #TODO why do I need to do this for KMeans, but not GMM?
   copy!(km.means, _km.means)

   return _ll, alpha_k
end

# }}}


## KMeans - Nesterov's 2nd method ##
# {{{
"""
Nesterov's 2nd method.  Accelerated gradient descent driver
"""
function nest2!{T}(
      km::KMeans{T},
      X::Array{T,2};
      n_iter::Int=25,
      n_em_iter::Int=2,
      ll_tol::T=1e-3,
      print=false)
   
   n_dim, n_clust, n_ex = data_sanity(km, X)
   
   # initially do a few EM steps
   prev_ll = -T(Inf)
   for it in 1:n_em_iter
      prev_ll = em_step!(km, X)
      println("em!: log-likelihood = $(prev_ll)")
   end

   nu = deepcopy(km)
   y = deepcopy(km)

   ll_diff = T(0)
   bt_step = T(1)
   for it in 1:n_iter
      theta = T(2)/T(it+1)
      weighted_sum!(y, T(1)-theta, km, theta, nu)
  
      # naive step size
      #ll = nest2_step!(km, X, step_size=:em_step)
      #ll = gd_step!(km, X, step_size=1e-4/(1+it)^(.7))
      
      # backtracking line search
      if it == 1
         ll, bt_step = nest2_bt_ls_step!(nu, y, X, theta, alpha=bt_step)
      else
         #ll, bt_step = nest2_bt_ls_step!(nu, y, X, theta)
         #ll, bt_step = nest2_bt_ls_step!(nu, y, X, theta, alpha=bt_step)
         # hacky way to attempt to grow step size
         ll, bt_step = nest2_bt_ls_step!(nu, y, X, theta, alpha=2.0*bt_step)
      end

      weighted_sum!(km, T(1)-theta, km, theta, nu)

      if print
         println("nest2!: log-likelihood = $(ll)")
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
      warn("Log-likelihood has not reached convergence criterion of $(ll_tol) in $(n_iter) iterations.  Nesterov's 2nd method may not have converged!")

   else
      km.trained = true
   end

   return km
end


"""
Form a weighted sum of 
km <- alpha*km1 + beta*km2
"""
function weighted_sum!{T}(
      km::KMeans{T},
      alpha::T, km1::KMeans{T},
      beta::T, km2::KMeans{T})
   
   size(km.means) == size(km1.means) == size(km2.means) ||
      error("Number of clusters for weighted sum should be the same.")

   size(km.means[1]) == size(km1.means[1]) == size(km2.means[2]) ||
      error("Number of dimensions for weighted sum should be the same.")

   for k in 1:size(km.means,1)
      km.means[k] = alpha*km1.means[k] + beta*km2.means[k]
   end
end


"""
simple backtracking line search method for Nesterov's second method
`nu` - mix of current x and previous nu
`y` - mix of current x and current nu, used for gradient
`X` - data matrix (n_ex, n_dim)
`theta_k` - 2/(k+1) for nest2
`alpha` - step size to start with
`rho` - backtracking step size reduction factor
`c` - sufficient step criterion
"""
function nest2_bt_ls_step!{T<:Real}(
      nu::KMeans{T},
      y::KMeans{T},
      X::Array{T,2},
      theta_k::T;
      alpha::T=1e0,
      rho::T=0.5,
      c::T=1e-4)
   
   function step_km!(km, mean_grad, alpha)
      n_dim, n_clust, n_ex = data_sanity(km, X)

      for k in 1:n_clust
         km.means[k] += alpha*mean_grad[k]
      end
   end
   
   n_dim, n_clust, n_ex = data_sanity(nu, X)
   ll, mean_grad, resp  = compute_grad(y, X)
   
   alpha_k = alpha
   _nu = deepcopy(nu)
   step_km!(_nu, mean_grad, alpha_k/theta_k)
   _ll = compute_ll(_nu, X)

   grad_ip = T(0)
   for k in 1:n_clust
      grad_ip += sumabs2(mean_grad[k])
   end
   #println("  grad_ip = $(grad_ip)")
   
   #TODO convergence crit?
   crit = T(0)
   for k in 1:n_clust
      crit += dot(mean_grad[k], _nu.means[k]-y.means[k])
      crit -= T(theta_k^2/(2.0*alpha_k))*sumabs2(_nu.means[k]-y.means[k])
   end
   
   ls_count = 1
   #println(c*alpha_k/theta_k*grad_ip)
   #while isnan(_ll) || _ll < ll + c*alpha_k*grad_ip
   #while isnan(_ll) || _ll < ll + c*alpha_k/theta_k*grad_ip
   while isnan(_ll) || _ll < ll + c*crit
      alpha_k *= rho 

      _nu = deepcopy(nu)
      step_km!(_nu, mean_grad, alpha_k/theta_k)
      _ll = compute_ll(_nu, X)
      #println("  alpha_k = $(alpha_k), _ll = $(_ll), diff = $(_ll-ll-c*alpha_k*grad_ip)")

      crit = T(0)
      for k in 1:n_clust
         crit += dot(mean_grad[k], _nu.means[k]-y.means[k])
         crit -= T(theta_k^2/(2.0*alpha_k))*sumabs2(_nu.means[k]-y.means[k])
      end

      if alpha_k < eps(ll)
         return ll, alpha_k # sufficient decrease not found
      end

      ls_count += 1
      #println("ls_count = $(ls_count)")
   end
   
   #TODO why do I need to do this for k-means, but not GMM?
   copy!(nu.means, _nu.means)

   return _ll, alpha_k
  
end

# }}}


## Gradient ##
# {{{
"""
Gradient for KMeans
"""
function compute_grad{T}(
      km::KMeans{T},
      X::Array{T,2})
   
   n_dim, n_clust, n_ex = data_sanity(km, X)
   
   sigma = T(1)

   wrk = Array{T}(n_ex, n_dim)
   resp = Array{T}(n_ex, n_clust)
   ll = T(0)
   
   for k = 1:n_clust
      broadcast!(-, wrk, X, km.means[k].')
      wrk .*= wrk
      resp[:,k] = -sum(wrk, 2)/(2*sigma^2)

   end

   # log-sum-exp trick
   m = maximum(resp,2)
   broadcast!(-, resp, resp, m)
   resp = exp(resp)
   ll = (sum(m) + sum(log(sum(resp,2))) - log(T(n_clust)))/T(n_ex)

   # normalize
   # Baye's rule/softmax to normalize responsibilities
   broadcast!(/, resp, resp, sum(resp, 2))

   # sometimes zero-sum responsibilities are introduced (at least for GMMs)
   resp[find(isnan(resp))] = T(0) # set NaNs to zero
   
   rk = sum(resp, 1)       # \sum_{i=1}^N r_{ik}
   rik_X = resp.'*X        # used for \sum_{i=1}^N r_{ik} x_i
   mean_grad = Array{Array{T,1},1}(n_clust)
   for k = 1:n_clust
      mean_grad[k] = -T(1)/sigma*rk[k]*km.means[k]
      mean_grad[k] += vec(T(1)/sigma^2*rik_X[k,:])
   end

   return ll, mean_grad, resp
end

# }}}

