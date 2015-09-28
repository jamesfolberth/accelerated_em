
using Base.LinAlg.BLAS: ger!

# EM algorithm and helpers
function em!{T<:AbstractFloat}(
      gmm::GMM{T},
      X::Array{T,2};
      n_iter::Int=25,
      ll_tol::T=1e-4,
      print=false)

   n_ex, n_dim = size(X)
   k = size(gmm.means, 1) # num components
      
   prev_ll = -T(Inf)
   ll_diff = T(0.0)
   for it in 1:n_iter
      ll = em_step!(gmm, X)

      if print
         println("log-likelihood = $(ll)")
      end
      
      ll_diff = abs(prev_ll - ll)
      if ll_diff < ll_tol
         break
      end
      prev_ll = ll
   end
   
   if ll_diff > ll_tol
      warn("Log-likelihood has not reached convergence criterion of $(ll_tol) in $(n_iter) iterations.  EM may not have converged!")

   else
      gmm.trained = true
   end

   return gmm
end


function em_step!{T<:AbstractFloat}(
      gmm::GMM{T},
      X::Array{T,2};
      var_thresh::T=1e-3,
      chol_thresh::T=1e-3)

   n_ex, n_dim = size(X)
   k = size(gmm.means, 1) # num components

   if gmm.cov_type == :diag
      # E step
      # compute responsibilities
      prec = map(cm->T(1)./cm.diag, gmm.covs)
      cov_det = map(cm->prod(cm.diag), gmm.covs) # det(Sigma)

      wrk = Array{T}(n_ex, n_dim) # used for storing (X-mean)
      logpdf = Array{T}(n_ex, k)  # log p(y_i | mean_k, Sigma_k)
      resp = Array{T}(n_ex, k)    # responsibility of component j for example i
      for j in 1:k
         broadcast!(-,wrk,X,gmm.means[j].')     # (x-mean)
         broadcast!(.*,wrk,wrk.*wrk, prec[j].') # (x-mean)^T*prec*(x-mean)
         logpdf[:,j] = -T(0.5)*sum(wrk, 2) - T(0.5)*log(cov_det[j]) - T(n_dim*0.5)*log(2*pi)
         resp[:,j] = gmm.weights[j]*exp(logpdf[:,j])
      end

      logpdf = copy(resp) # will use later for log-likelihood

      # Baye's rule to normalize responsibilities
      broadcast!(/, resp, resp, sum(resp, 2))

      # M step
      rk = sum(resp, 1)       # \sum_{i=1}^N r_{ik}
      rik_X = resp.'*X        # used for \sum_{i=1}^N r_{ik} x_i
      rik_XXT = resp.'*(X.*X) # used for \sum_{i=1}^N r_{ik} x_i x_i^T
      for j in 1:k
         # updates
         gmm.weights[j] = rk[j]/T(n_ex)
         gmm.means[j] = vec(rik_X[j,:])/rk[j]
         gmm.covs[j].diag[:] = vec(rik_XXT[j,:])/rk[j] - gmm.means[j].^2

         # threshold the variances
         for ind in find(gmm.covs[j].diag .< var_thresh)
            gmm.covs[j].diag[ind] = var_thresh
         end
      end

      # return log-likelihood
      # here logpdf is just used as a work array to store resp before
      # Baye's normalization
      ll = reshape(sum(log(sum(logpdf, 2)), 1)/T(n_ex), 1)[1]
      return ll

   elseif gmm.cov_type == :full
      # E step
      # compute responsibilities
      cov_logdet = map(cm->logdet(cm.chol), gmm.covs) # logdet(Sigma)

      wrk = Array{T}(n_dim, n_ex) # used for storing (X-mean)
      logpdf = Array{T}(n_ex, k)  # log p(y_i | mean_k, Sigma_k)
      resp = Array{T}(n_ex, k)    # responsibility of component j for example i
      for j in 1:k
         broadcast!(-,wrk,X.',gmm.means[j])     # (x-mean)
         wrk = gmm.covs[j].chol[:L] \ wrk       # L^{-1}*(x-mean)
         wrk .*= wrk                            # (x-mean)^T*prec*(x-mean)
         logpdf[:,j] = -T(0.5)*sum(wrk, 1) - T(0.5)*cov_logdet[j] - T(n_dim*0.5)*log(2*pi)
         resp[:,j] = gmm.weights[j]*exp(logpdf[:,j])
      end

      logpdf = copy(resp) # will use later for log-likelihood

      # Baye's rule to normalize responsibilities
      broadcast!(/, resp, resp, sum(resp, 2))
      
      # M step
      rk = sum(resp, 1)       # \sum_{i=1}^N r_{ik}
      rik_X = resp.'*X        # used for \sum_{i=1}^N r_{ik} x_i
      
      rik_XXT = Array{Array{T,2},1}(k) # used for \sum_{i=1}^N r_{ik} x_i x_i^T
      for j in 1:k
         rik_XXT[j] = Array{T}(zeros(n_dim, n_dim))
      end
      @inbounds for i in 1:n_ex
         x = vec(X[i,:])
         for j in 1:k
            ger!(resp[i,j], x, x, rik_XXT[j]) 
         end
      end
      
      for j in 1:k
         # updates
         gmm.weights[j] = rk[j]/T(n_ex)
         gmm.means[j] = vec(rik_X[j,:])/rk[j]
         gmm.covs[j].cov = rik_XXT[j]/rk[j]
         ger!(-T(1.0), gmm.means[j], gmm.means[j], gmm.covs[j].cov)

         # threshold the covariance matrices?
         #TODO
         try
            gmm.covs[j].chol = cholfact(gmm.covs[j].cov)
         catch
            println(gmm.covs[j].cov)
            error("Covariances have gone funky!")
         end
      end

      # return log-likelihood
      # here logpdf is just used as a work array to store resp before
      # Baye's normalization
      ll = reshape(sum(log(sum(logpdf, 2)), 1)/T(n_ex), 1)[1]
      return ll
   
   else
      error("Unknown covariance type $(gmm.cov_type).")
   end

end


