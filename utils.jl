
## Some GMM utilitites ##
# score_samples like sklearn's GMM class
# https://github.com/scikit-learn/scikit-learn/blob/a95203b/sklearn/mixture/gmm.py#L285
"""
Return the per sample likelihood under the GMM

Args:
    gmm - GMM that has been fitted
    X - sample data of size n_ex * n_dim

Returns:
    scores - per sample likelihood of size n_ex
    resp - responsibilities for sample i from component j 
"""
function score_samples{T<:AbstractFloat}(gmm::GMM{T}, X::Array{T,2})
   if ! (gmm.trained)
      error("You must train the GMM before scoring.")
   end
   
   n_ex, n_dim = size(X)
   k = size(gmm.weights,1)

   if gmm.cov_type == :diag
      # This is basically the E-step of EM
      prec = map(cm->T(1)./cm.diag, gmm.covs)
      cov_det = map(cm->prod(cm.diag), gmm.covs) # det(Sigma)

      wrk = Array{T}(n_ex, n_dim) # used for storing (X-mean)
      resp = Array{T}(n_ex, k)  # log p(y_i | mean_k, Sigma_k)
      for j in 1:k
         broadcast!(-,wrk,X,gmm.means[j].')     # (x-mean)
         broadcast!(.*,wrk,wrk.*wrk, prec[j].') # (x-mean)^T*prec*(x-mean)
         resp[:,j] = -T(0.5)*sum(wrk, 2) - T(0.5)*log(cov_det[j]) - T(n_dim*0.5)*log(2*pi)
         resp[:,j] = gmm.weights[j]*exp(resp[:,j])
      end

      return sum(resp,2), resp # sum over components

   else
      error("Unsupported covariance type $(gmm.cov_type)")
   end

end

## Plotting ##
function plot_data(X, y)

   if size(X, 2) == 1
      error("Not implemented.")

   elseif size(X, 2) == 2
      scatter(X[:,1], X[:,2], c=y)
      xlabel("x")
      ylabel("y")
      axis("equal")

   elseif size(X, 2) == 3
      error("Not implemented.")

   else
      error("Plotting data in $(size(X,2)) dimensions not supported.")

   end
end

function plot_data(X)
   y = ones(Int, size(X,1))
   plot_data(X,y)
end


function plot_gmm_contours{T<:AbstractFloat, B<:Real}(
      gmm::GMM{T},
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

function plot_gmm_contours{T<:AbstractFloat, B<:Real}(
      gmm::GMM{T},
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


