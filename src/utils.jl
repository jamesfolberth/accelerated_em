
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


# mean silhoette coefficient
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#sklearn.metrics.silhouette_score
# https://en.wikipedia.org/wiki/Silhouette_(clustering)
function silhoette_score{T<:Real,S<:Integer}(X::Array{T,2}, y::Array{S,1})

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

   length(keys(cluster_inds)) >= 2 || error("Similarity score not defined if number of clusters < 2.")
   length(keys(cluster_inds)) <= n_ex -1 || error("Similarity score not defined if number of" *
      " clusters is greater than n_ex-1")

   # compute full distance matrix
   # this is more efficient than the obvious implementation when X is n_ex x n_dim
   dist = Array{T}(zeros(n_ex, n_ex))
   for k in 1:n_dim
      for i in 1:n_ex
         for j in 1:n_ex
            dist[i,j] += (X[i,k]-X[j,k])^2
         end
      end
   end
   for i in 1:n_ex
      dist[i,i] = sqrt(dist[i,i])
      for j in i+1:n_ex
         dist[i,j] = sqrt(dist[i,j])
         dist[j,i] = dist[i,j]
      end
   end
   
   # mean intra-cluster dissimilarity
   a = Array{T}(n_ex)
   # min nearest cluster mean dissimilarity
   b = Array{T}(n_ex)
   tmp = Array{T}(length(keys(cluster_inds))-1)
   for j in 1:n_ex
      a[j] = mean(dist[cluster_inds[y[j]],j])
      
      ind = 0
      for c in keys(cluster_inds)
         if c == y[j]
            continue
         else
            tmp = mean(dist[cluster_inds[c],j]) 
            ind += 1
         end
      end
      b[j] = minimum(tmp)
   end
   
   sc = (b-a) ./ max(a,b)

   return mean(sc)
end


# standardize data so that each dimension has mean 0 and variance 0
function standard_scaler!{T<:Real}(X::Array{T,2}; variance=true)
   
   if variance
      println("Variance = true")
      m = mean(X,1)
      s = stdm(X,m,1)
      ep = 1e2*eps(mean(m))
      s[s .< ep] = T(0) # if std dev is too small
      broadcast!(-, X, X, m)
      broadcast!(/, X, X, s)
      return m, s

   else # just shift the mean
      m = mean(X,1)
      broadcast!(-, X, X, m)
      return m
   end
end

function standard_scaler{T<:Real}(X::Array{T,2}; variance=true)
   
   Xc = copy(X)
   if variance
      m, s = standard_scaler!(Xc, variance=variance)
      return Xc, m, s

   else
      m = standard_scaler!(Xc, variance=variance)
      return Xc, m
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

