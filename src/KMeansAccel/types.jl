
type KMeans{T}
   n_dim::Integer               # dimensionality of data
   n_clust::Integer             # number of clusters
   means::Array{Array{T,1},1}   # 1d array of 1d mean arrays
   init::Bool                   # is the GMM initialized and ready to be fit?
   trained::Bool                # is the GMM fit to data?
end


## KMeans constructors ##
function KMeans{T}(
   X::Array{T,2};
   K::Int=3,
   mean_init_method=:kmpp)
   
   n_ex, n_dim = size(X)
   (n_ex >= K && K > 0) || throw("Number of examples cannot be less than K")

   # initialize means
   means = Array{Array{T, 1}, 1}(K)
   if mean_init_method == :zeros
      fill!(means, zeros(T, n_dim))

   elseif mean_init_method == :rand
      for ind in 1:K
         means[ind] = Array{T}(randn(n_dim))
      end

   elseif mean_init_method == :rand_centers
      inds = randperm(n_ex)[1:K]
      for (i, ind) in enumerate(inds)
         means[i] = vec(X[ind,:])
      end

   elseif mean_init_method == :kmpp
      # k-means++: The Advantages of Careful Seeding
      # David Arthur and Sergei Vassilvitskii

      # pick first center randomly
      ind = rand(1:n_ex)
      means[1] = vec(X[ind,:])

      if K > 1
         
         # squared distance to closest mean already selected,
         # then normalized so it's a probability
         dist_prob = Array{T}(zeros(n_ex))
         @inbounds for n in 1:n_ex
            dist_prob[n] = norm(X[n,:]-X[ind,:],2)^2
         end
         dist_prob[ind] = T(0)
         dist_prob /= sum(dist_prob)
         
         tmp = Array{T}(zeros(n_ex))
         for k=2:K

            # find a new center according to dist_prob
            cumsum!(tmp, dist_prob)
            ind = searchsortedfirst(tmp, rand())
            means[k] = vec(X[ind,:])
            
            # update distances
            @inbounds for n in 1:n_ex
               d = norm(X[n,:]-X[ind,:],2)^2
               if d < dist_prob[n]
                  dist_prob[n] = d
               end
            end
            dist_prob[ind] = T(0)
            dist_prob /= sum(dist_prob)
         end
      end

   else
      error("Unknown mean initialization method $(mean_init_method)")
   end

   return KMeans{T}(n_dim, K, means, true, false)
end


## make KMeans print a bit prettier
function Base.show{T}(io::IO, kmeans::KMeans{T})
   println(io, string("K-means model in $(kmeans.n_dim) ",
      "dimensions with $(kmeans.n_clust) clusters."))

   println(io, "  initialized: $(kmeans.init)")
   println(io, "  trained:     $(kmeans.trained)")

   for k in 1:kmeans.n_clust
      println(io, "  Component $(k):")
      println(io, "    mean:")
      pretty_print_vector(kmeans.means[k], indent_level=6)
   end
end
 
