
## Some GMM utilitites ##

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

"""
Verify that dimensions of KMeans and data matrix X make sense
"""
function data_sanity{T}(km::KMeans{T}, X::Array{T,2})
   n_dim = km.n_dim
   n_clust = km.n_clust
   n_ex = size(X,1)
   if n_dim != size(X,2)
      throw(DimensionMismatch(string("X has second dimension $(size(X,2)), but",
         " should match dimension of KMeans, $(n_dim)")))   
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


## KMeans make assignments
function hard_classify{T}(km::KMeans{T}, X::Array{T,2})
   
   km.trained || throw("km should be trained before using it for classificaiton.")
   n_dim, n_clust, n_ex = data_sanity(km, X)
   Xt = X.' # make a copy for local use

   assignments = zeros(Int64, n_ex)
   d = zeros(n_clust, n_ex)
  
   # update assignments
   for j in 1:n_ex
      Xj = vec(Xt[:,j])
      for i = 1:n_clust
         d[i,j] = norm(km.means[i] - Xj,2)
      end
      assignments[j] = indmin(d[:,j])
   end

   return assignments
end

function soft_classify{T}(km::KMeans{T}, X::Array{T,2})
   
   km.trained || throw("km should be trained before using it for classificaiton.")
   n_dim, n_clust, n_ex = data_sanity(km, X)
   
   nrms = Array{T}(n_clust)
   assignments = Array{Int64}(n_ex)
   
   @inbounds for i in 1:n_ex
      Xi = vec(X[i,:])
      for k in 1:n_clust
         nrms[k] = norm(Xi-km.means[k],2)
      end
      assignments[i] = indmin(nrms)
   end
   
   return assignments
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


function plot_means{T}(km::KMeans{T})
   
   ax = gca()
   if km.n_dim == 1
      error("not implemented")

   elseif km.n_dim == 2
      for mean in km.means
         c = mpatches.Circle((mean[1],mean[2]), 1)
         ax[:add_patch](c)
      end
      show()

   elseif km.n_dim == 3
      error("not implemented")

   else
      error("Plotting in $(km.n_dim) dimensions not supported.")
   end

end


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


## Pretty printing ##
function pretty_print_vector{T}(io::IO, x::AbstractArray{T}; indent_level::Integer=0)
   for val in x
      println(io, join([repeat(" ",indent_level), @sprintf "% 7.3f" val]))
   end
end
#TODO: is there a way to clean this up?  Like python's dict unpacking with param'd types?
#      one constraint is that we want to mimic the call to println which has io optional in front
pretty_print_vector{T}(x::AbstractArray{T}; indent_level::Integer=0) = 
   pretty_print_vector(STDOUT, x, indent_level=indent_level)

function pretty_print_matrix{T}(io::IO, mat::AbstractArray{T,2}; indent_level::Integer=0)
   for i in 1:size(mat, 1)
      print(io, repeat(" ", indent_level))
      for val in mat[i,:]
         print(io, @sprintf "% 7.3f  " val)
      end
      println(io,"")
   end
end
pretty_print_matrix{T}(mat::AbstractArray{T,2}; indent_level::Integer=0) = 
   pretty_print_matrix(STDOUT, mat, indent_level=indent_level)

