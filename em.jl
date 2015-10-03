
using Base.LinAlg.BLAS: ger!

# EM algorithm and helpers
function em!{T,CM<:CovMat}(
      gmm::GMM{T,CM},
      X::Array{T,2};
      n_iter::Int=25,
      ll_tol::T=1e-3,
      print=false)

   n_dim, n_clust, n_ex = gmm_data_sanity(gmm, X)
      
   prev_ll = -T(Inf)
   ll_diff = T(0.0)
   for it in 1:n_iter
      #println(gmm.cov_type)
      #println(typeof(gmm))
      ll = em_step!(gmm, X)

      ## testing compute_gradient
      #println("compute_gradient:")
      #(_, weight_grad, mean_grad, chol_grad) = compute_gradient(gmm, X)
      #println(mean_grad[1])
      ##println(chol_grad[1])

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
      gmm.trained = true
   end

   return gmm
end


## EM steppers ##
function em_step!{T,CM<:DiagCovMat}(
      gmm::GMM{T,CM},
      X::Array{T,2};
      var_thresh::T=1e-3)

   n_dim, n_clust, n_ex = gmm_data_sanity(gmm, X)

   ## E step ##
   # compute responsibilities
   prec = map(cm->T(1)./cm.diag, gmm.covs)
   cov_det = map(cm->prod(cm.diag), gmm.covs) # det(Sigma)

   wrk = Array{T}(n_ex, n_dim) # used for storing (X-mean)
   logpdf = Array{T}(n_ex, n_clust)  # log p(y_i | mean_k, Sigma_k)
   resp = Array{T}(n_ex, n_clust)    # responsibility of component j for example i
   for k in 1:n_clust
      broadcast!(-,wrk,X,gmm.means[k].')     # (x-mean)
      broadcast!(.*,wrk,wrk.*wrk, prec[k].') # (x-mean)^T*prec*(x-mean)
      logpdf[:,k] = -T(0.5)*sum(wrk, 2) - T(0.5)*log(cov_det[k]) -
         T(n_dim*0.5)*log(2*pi)
      resp[:,k] = gmm.weights[k]*exp(logpdf[:,k])
   end

   logpdf = copy(resp) # will use later for log-likelihood

   # Baye's rule to normalize responsibilities
   broadcast!(/, resp, resp, sum(resp, 2))

   # sometimes zero-sum responsibilities are introduced
   resp[find(isnan(resp))] = T(0.0) # set NaNs to zero


   ## M step ##
   rk = sum(resp, 1)       # \sum_{i=1}^N r_{ik}
   rik_X = resp.'*X        # used for \sum_{i=1}^N r_{ik} x_i
   rik_XXT = resp.'*(X.*X) # used for \sum_{i=1}^N r_{ik} x_i x_i^T
   for k in 1:n_clust
      # updates
      gmm.weights[k] = rk[k]/T(n_ex)
      gmm.means[k] = vec(rik_X[k,:])/rk[k]
      gmm.covs[k].diag[:] = vec(rik_XXT[k,:])/rk[k] - gmm.means[k].^2

      # threshold the variances
      for ind in find(gmm.covs[k].diag .< var_thresh)
         gmm.covs[k].diag[ind] = var_thresh
         warn("Covariances have gone funky, attempting to force PD!")
      end
   end

   # return log-likelihood
   # here logpdf is just used as a work array to store resp before
   # Baye's normalization
   ll = reshape(sum(log(sum(logpdf, 2)), 1)/T(n_ex), 1)[1]
   return ll
end

function em_step!{T,CM<:FullCovMat}(
      gmm::GMM{T,CM},
      X::Array{T,2};
      chol_thresh::T=1e-1)

   n_dim, n_clust, n_ex = gmm_data_sanity(gmm, X)

   ## E step ##
   # compute responsibilities
   cov_logdet = map(cm->logdet(cm.chol), gmm.covs) # logdet(Sigma)

   wrk = Array{T}(n_dim, n_ex) # used for storing (X-mean)
   logpdf = Array{T}(n_ex, n_clust)  # log p(y_i | mean_k, Sigma_k)
   resp = Array{T}(n_ex, n_clust)    # responsibility of component j for example i
   for k in 1:n_clust
      broadcast!(-,wrk,X.',gmm.means[k])     # (x-mean)
      wrk = gmm.covs[k].chol[:L] \ wrk       # R^{-T}*(x-mean)
      wrk .*= wrk                            # (x-mean)^T*prec*(x-mean)
      logpdf[:,k] = -T(0.5)*sum(wrk, 1) - T(0.5)*cov_logdet[k] -
         T(n_dim*0.5)*log(2*pi)
      resp[:,k] = gmm.weights[k]*exp(logpdf[:,k])
   end

   logpdf = copy(resp) # will use later for log-likelihood

   # Baye's rule to normalize responsibilities
   broadcast!(/, resp, resp, sum(resp, 2))
   
   # sometimes zero-sum responsibilities are introduced
   resp[find(isnan(resp))] = T(0.0) # set NaNs to zero
   

   ## M step ##
   rk = sum(resp, 1)       # \sum_{i=1}^N r_{ik}
   rik_X = resp.'*X        # used for \sum_{i=1}^N r_{ik} x_i
   
   rik_XXT = Array{Array{T,2},1}(n_clust) # used for \sum_{i=1}^N r_{ik} x_i x_i^T
   for k in 1:n_clust
      rik_XXT[k] = Array{T}(zeros(n_dim, n_dim))
   end
   @inbounds for i in 1:n_ex
      x = vec(X[i,:])
      for k in 1:n_clust
         ger!(resp[i,k], x, x, rik_XXT[k]) 
      end
   end
   
   for k in 1:n_clust
      # updates
      gmm.weights[k] = rk[k]/T(n_ex)
      gmm.means[k] = vec(rik_X[k,:])/rk[k]
      gmm.covs[k].cov = rik_XXT[k]/rk[k]
      ger!(-T(1.0), gmm.means[k], gmm.means[k], gmm.covs[k].cov)

      # threshold the covariance matrices
      #TODO what's a reasonable way to do this?
      try
         gmm.covs[k].chol = cholfact(gmm.covs[k].cov)
      catch
         warn("Covariance $(k) has gone funky, attempting to force PD!")
         gmm.covs[k].cov += chol_thresh*Array{T}(eye(n_dim))
         try
            gmm.covs[k].chol = cholfact(gmm.covs[k].cov)
         catch
            warn(string("Covariance $(k) is super funky, reverting to prior ",
               "iterate's covariance.!"))
            cf = gmm.covs[k].chol
            gmm.covs[k].cov = cf[:L]*cf[:U]
         end
      end
   end

   # return log-likelihood
   # here logpdf is just used as a work array to store resp before
   # Baye's normalization
   ll = reshape(sum(log(sum(logpdf, 2)), 1)/T(n_ex), 1)[1]
   return ll
end


## log-likelihood ##
function compute_ll{T,CM<:DiagCovMat}(
      gmm::GMM{T,CM},
      X::Array{T,2})

   n_dim, n_clust, n_ex = gmm_data_sanity(gmm, X)
   
   # this is similar to E step
   prec = map(cm->T(1)./cm.diag, gmm.covs)
   cov_det = map(cm->prod(cm.diag), gmm.covs) # det(Sigma)

   wrk = Array{T}(n_ex, n_dim) # used for storing (X-mean)
   logpdf = Array{T}(n_ex, n_clust)  # log p(y_i | mean_k, Sigma_k)
   for k in 1:n_clust
      broadcast!(-,wrk,X,gmm.means[k].')     # (x-mean)
      broadcast!(.*,wrk,wrk.*wrk, prec[k].') # (x-mean)^T*prec*(x-mean)
      logpdf[:,k] = -T(0.5)*sum(wrk, 2) - T(0.5)*log(cov_det[k]) -
         T(n_dim*0.5)*log(2*pi)
      logpdf[:,k] = gmm.weights[k]*exp(logpdf[:,k]) # no longer logpdf
   end

   # return log-likelihood
   ll = reshape(sum(log(sum(logpdf, 2)), 1)/T(n_ex), 1)[1]
   return ll
end

function compute_ll{T,CM<:FullCovMat}(
      gmm::GMM{T,CM},
      X::Array{T,2})

   n_dim, n_clust, n_ex = gmm_data_sanity(gmm, X)
   
   # this is similar to E step
   cov_logdet = map(cm->logdet(cm.chol), gmm.covs) # logdet(Sigma)

   wrk = Array{T}(n_dim, n_ex) # used for storing (X-mean)
   logpdf = Array{T}(n_ex, n_clust)  # log p(y_i | mean_k, Sigma_k)
   for k in 1:n_clust
      broadcast!(-,wrk,X.',gmm.means[k])     # (x-mean)
      wrk = gmm.covs[k].chol[:L] \ wrk       # R^{-T}*(x-mean)
      wrk .*= wrk                            # (x-mean)^T*prec*(x-mean)
      logpdf[:,k] = -T(0.5)*sum(wrk, 1) - T(0.5)*cov_logdet[k] -
         T(n_dim*0.5)*log(2*pi)
      logpdf[:,k] = gmm.weights[k]*exp(logpdf[:,k]) # no longer logpdf
   end
   
   # return log-likelihood
   ll = reshape(sum(log(sum(logpdf, 2)), 1)/T(n_ex), 1)[1]
   return ll
end

