
# EM algorithm and helpers
"""
Train using standard K-means algorithm
"""
function hard_em!{T}(
      km::KMeans{T},
      X::Array{T,2};
      n_iter::Int=25,
      print=false)
   
   n_dim, n_clust, n_ex = data_sanity(km, X)
   Xt = X.' # make a copy for local use
   
   means_to_update = Array{Bool}(n_clust)
   fill!(means_to_update, true)
   clust_counts = zeros(T, n_clust)
   assignments = zeros(Int64, n_ex)

   # matrix for squared distances
   d2 = zeros(n_clust, n_ex)
   
   n_it = 0
   # iterate until assignments don't change
   # could instead iterate until change in LL is small enough
   while any(means_to_update)
      n_it += 1

      # update assignments
      fill!(means_to_update, false)
      for j in 1:n_ex
         Xj = vec(Xt[:,j])
         for i = 1:n_clust
            d2[i,j] = norm(km.means[i] - Xj,2)^2
         end
         aj = indmin(d2[:,j])
         if aj != assignments[j]
            means_to_update[aj] = true
            assignments[j] = aj
         end
      end
 
      # update means
      fill!(clust_counts, T(0))
      fill!(km.means, zeros(T, n_dim))
      for j in 1:n_ex
         a = assignments[j]
         km.means[a] += vec(Xt[:,j])
         clust_counts[a] += T(1)
      end
      for i in 1:n_clust
         km.means[i] /= clust_counts[i]
      end
   
   end

   km.trained = true
   
end

"""
Train using soft K-means algorithm
"""
function em!{T}(
      km::KMeans{T},
      X::Array{T,2};
      n_iter::Int=25,
      ll_tol::T=1e-3,
      print=false)

   n_dim, n_clust, n_ex = data_sanity(km, X)
      
   prev_ll = -T(Inf)
   ll_diff = T(0.0)
   for it in 1:n_iter
      ll = em_step!(km, X)

      if print
         println("em!: log-likelihood = $(ll)")
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
      warn(string("Log-likelihood has not reached convergence criterion of ",
         "$(ll_tol) in $(n_iter) iterations.  EM may not have converged!"))

   else
      km.trained = true
   end

   return km
end

"""
Soft k-means mean update.
"""
function em_step!{T}(
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
   for k = 1:n_clust
      km.means[k] = vec(rik_X[k,:])/rk[k]
   end

   return ll
end


"""
Compute log-likelihood for current parameters
"""
function compute_ll{T}(
      km::KMeans{T},
      X::Array{T,2})

   n_dim, n_clust, n_ex = data_sanity(km, X)
   
   # this is the E step for soft k-means,
   # but should also hold for hard k-means, as it's only the M
   # step that is different
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
 
   # the rest of the E step would normalize resp
   return ll
end

