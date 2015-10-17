
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


# make assignments
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
# {{{

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



# }}}
