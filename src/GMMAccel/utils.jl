
"""
Verify that dimensions of GMM and data matrix X make sense
"""
function data_sanity{T,CM<:CovMat}(gmm::GMM{T,CM}, X::Array{T,2})
   n_dim = gmm.n_dim
   n_clust = gmm.n_clust
   n_ex = size(X,1)
   if n_dim != size(X,2)
      throw(DimensionMismatch(string("X has second dimension $(size(X,2)), but",
         " should match dimension of GMM, $(n_dim)")))   
   end

   return n_dim, n_clust, n_ex 
end


# score_samples like sklearn's GMM class
# https://github.com/scikit-learn/scikit-learn/blob/a95203b/sklearn/mixture/gmm.py#L285
"""
Return the per sample likelihood under the GMM

Args:
    gmm - GMM that has been fitted
    X - sample data of size n_ex * n_dim

Returns:
    scores - per sample likelihood of size n_ex
    resp - responsibilities for sample i from component k
"""
function score_samples{T,CM<:DiagCovMat}(gmm::GMM{T,CM}, X::Array{T,2})
   if ! (gmm.trained)
      error("You must train the GMM before scoring.")
   end
   
   n_dim, n_clust, n_ex = data_sanity(gmm, X)

   # This is basically the E-step of EM
   prec = map(cm->T(1)./cm.diag, gmm.covs)
   cov_det = map(cm->prod(cm.diag), gmm.covs) # det(Sigma)

   wrk = Array{T}(n_ex, n_dim) # used for storing (X-mean)
   resp = Array{T}(n_ex, n_clust)  # log p(y_i | mean_k, Sigma_k)
   for k in 1:n_clust
      broadcast!(-,wrk,X,gmm.means[k].')     # (x-mean)
      broadcast!(.*,wrk,wrk.*wrk, prec[k].') # (x-mean)^T*prec*(x-mean)
      resp[:,k] = -T(0.5)*sum(wrk, 2) - T(0.5)*log(cov_det[k]) - T(n_dim*0.5)*log(2*pi)
      resp[:,k] = gmm.weights[k]*exp(resp[:,k])
   end

   return sum(resp,2), resp # sum over components
end

function score_samples{T,CM<:FullCovMat}(gmm::GMM{T,CM}, X::Array{T,2})
   if ! (gmm.trained)
      error("You must train the GMM before scoring.")
   end
   
   n_dim, n_clust, n_ex = data_sanity(gmm, X)

   # This is basically the E-step of EM
   cov_logdet = map(cm->logdet(cm.chol), gmm.covs) # logdet(Sigma)

   wrk = Array{T}(n_dim, n_ex) # used for storing (X-mean)
   logpdf = Array{T}(n_ex, n_clust)  # log p(y_i | mean_k, Sigma_k)
   resp = Array{T}(n_ex, n_clust)    # responsibility of component k for example i
   for k in 1:n_clust
      broadcast!(-,wrk,X.',gmm.means[k])     # (x-mean)
      wrk = gmm.covs[k].chol[:L] \ wrk       # L^{-1}*(x-mean)
      wrk .*= wrk                            # (x-mean)^T*prec*(x-mean)
      logpdf[:,k] = -T(0.5)*sum(wrk, 1) - T(0.5)*cov_logdet[k] - T(n_dim*0.5)*log(2*pi)
      resp[:,k] = gmm.weights[k]*exp(logpdf[:,k])
   end

   return sum(resp,2), resp # sum over components
end


## Plotting ## 
# {{{

function plot_gmm_contours{T,B<:Real,CM<:CovMat}(
      gmm::GMM{T,CM},
      bounds::Array{B,1};
      nx::Int=100,
      ny::Int=100)

   if size(gmm.means[1],1) != 2
      error("plot_gmm_contours shoulds be used only in 2 dimensions.")
   end
         
   x_min = bounds[1]; x_max = bounds[2]
   y_min = bounds[3]; y_max = bounds[4]

   xv = Array{T}(collect(linspace(x_min, x_max, nx)))
   yv = Array{T}(collect(linspace(y_min, y_max, ny)))
   xx = xv*ones(T, (1, ny))
   yy = ones(T, (nx, 1))*yv.'

   X = [vec(xx) vec(yy)]
   scores, resp = score_samples(gmm, X)

   scores = reshape(scores, size(xx))

   contour(xx, yy, scores, levels=collect(logspace(-6,2,20)))

end

function plot_gmm_contours{T,B<:Real,CM<:CovMat}(
      gmm::GMM{T,CM},
      bounds::Array{B,2};
      n_lin::Int=100)

   m,n = size(bounds)
   if m == 1 && n == 4
      Tbounds = Array{T}(vec(bounds))
   elseif m == 4 && n == 1
      Tbounds = Array{T}(vec(bounds))
   else
      error("Bad bounds array shape: $(size(bounds))")
   end

   plot_gmm_contours(gmm, Tbounds, n_lin=n_lin)
end



# }}}

