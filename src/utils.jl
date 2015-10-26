
# intra-cluster distances for each cluster
# distance is the distance from the mean/centroid
function cluster_dist{T<:Real,S<:Integer}(
      km::EMAccel.KMeans{T},
      X::Array{T,2},
      y::Array{S,1})
   
   n_ex, n_dim = size(X)
   n_ex == size(y,1) || error("size(X,1) and size(y,1) should be the same")
   
   # collect indexes corresponding to each cluster
   cluster_inds = Dict{eltype(y),Array{Int64,1}}()
   for (i,yi) in enumerate(y)
      if !(yi in keys(cluster_inds))
         cluster_inds[yi] = [i]
      else
         push!(cluster_inds[yi], i)
      end
   end

   dist = zeros(T,length(keys(cluster_inds)))
   for (ind, (key, inds)) in enumerate(cluster_inds)
      for i in inds
         dist[ind] += norm(vec(X[i,:])-km.means[key],2)
      end
   end
   
   return dist
end


# standardize data so that each dimension has mean 0 and variance 0
function standard_scaler!{T<:Real}(X::Array{T,2})
   
   m = mean(X,1)
   s = stdm(X,m,1)
   ep = 1e2*eps(mean(m))
   s[s .< ep] = T(0) # if std dev is too small
   broadcast!(-, X, X, m)
   broadcast!(/, X, X, s)
   
   return m, s
end

function standard_scaler{T<:Real}(X::Array{T,2})
   
   Xc = copy(X)
   m, s = standard_scaler!(Xc)
   return Xc, m, s
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

