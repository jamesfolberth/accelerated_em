
## KMeans - simple GD ##
# {{{
"""
Gradient descent driver
"""
function gd!{T}(
      km::KMeans{T},
      X::Array{T,2};
      n_iter::Int=25,
      n_em_iter::Int=0,
      ll_tol::T=1e-3,
      print=false)
   
   n_dim, n_clust, n_ex = data_sanity(km, X)
   
   # initially do a few EM steps
   prev_ll = -T(Inf)
   for it in 1:n_em_iter
      prev_ll = em_step!(km, X)
      if print
         println("em!: log-likelihood = $(prev_ll)")
      end
   end

   ll_diff = T(0)
   bt_step = T(1)
   it_count = 0
   for it in 1:n_iter
      # naive step size
      #ll = gd_step!(km, X, step_size=:em_step)
      #ll = gd_step!(km, X, step_size=1e-4/(1+it)^(.7))
      
      # backtracking line search
      #TODO
      if it == 1
         ll, bt_step = bt_ls_step!(km, X, alpha=bt_step)
      else
         #ll, bt_step = bt_ls_step!(km, X, alpha=bt_step)
         # hacky way to attempt to grow step size
         ll, bt_step = bt_ls_step!(km, X, alpha=8.0*bt_step)
      end
      
      it_count += 1

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

   return it_count
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
      c::T=1e-5)
   
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
      n_em_iter::Int=0,
      ll_tol::T=1e-3,
      print=false)
   
   n_dim, n_clust, n_ex = data_sanity(km, X)
   
   # initially do a few EM steps
   prev_ll = -T(Inf)
   for it in 1:n_em_iter
      prev_ll = em_step!(km, X)
      if print
         println("em!: log-likelihood = $(prev_ll)")
      end
   end

   nu = deepcopy(km)
   y = deepcopy(km)

   ll_diff = T(0)
   bt_step = T(1)
   it_count = 0
   for it in 1:n_iter
      theta = T(2)/T(it+1)
      weighted_sum!(y, T(1)-theta, km, theta, nu)
  
      # naive step size
      ll = nest2_step!(nu, y, X, theta, step_size=:em_step)
      
      # backtracking line search
      #if it == 1
      #   ll, bt_step = nest2_bt_ls_step!(nu, y, X, theta, alpha=bt_step)
      #else
      #   #ll, bt_step = nest2_bt_ls_step!(nu, y, X, theta)
      #   #ll, bt_step = nest2_bt_ls_step!(nu, y, X, theta, alpha=bt_step)
      #   # hacky way to attempt to grow step size
      #   ll, bt_step = nest2_bt_ls_step!(nu, y, X, theta, alpha=8.0*bt_step)
      #end

      weighted_sum!(km, T(1)-theta, km, theta, nu)
   
      it_count += 1

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

   return it_count
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
take step of Nesterov's second method
`nu` - mix of current x and previous nu
`y` - mix of current x and current nu, used for gradient
`X` - data matrix (n_ex, n_dim)
`theta_k` - 2/(k+1) for nest2
`step_size` - :em_step or number
"""
function nest2_step!{T<:Real}(
      nu::KMeans{T},
      y::KMeans{T},
      X::Array{T,2},
      theta_k::T;
      step_size=1e-4)
   
   sigma = T(1)
   
   n_dim, n_clust, n_ex = data_sanity(nu, X)
   ll, mean_grad, resp  = compute_grad(y, X)
   rk = sum(resp,1)
   
   sigma = T(1)
   
   if step_size == :em_step
      for k in 1:n_clust
         eta = sigma^2/rk[k] # this step size recovers EM
         nu.means[k] += eta/theta_k*mean_grad[k]
      end

   elseif step_size <: T
      for k in 1:n_clust
         nu.means[k] += step_size/theta_k*mean_grad[k]
      end

   else
      error("Bad step size $(step_size)")
   end

   return ll
  
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
      c::T=1e-6)
   
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

## Gradients ##
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

