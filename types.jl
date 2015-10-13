
using Base.LinAlg: Cholesky
using Clustering: kmeans

## Define GMM types ##
abstract CovMat{T}
eltype{T}(::CovMat{T}) = T

type DiagCovMat{T} <: CovMat{T}
   #TODO should we use DenseArray, AbstractArray, or something else here?  Performance difference?
   #TODO Should be subtype T <: Real or T <: AbstractFloat?  Depends on what cholfact can handle?
   diag::Array{T,1} # main diagonal of covariance matrix
end

type FullCovMat{T} <: CovMat{T}
   cov::Array{T,2} # full covariance matrix
   chol::Cholesky{T,Array{T,2}} # Cholesky factorization type of cov
end
FullCovMat{T}(cov::Array{T,2}) = FullCovMat(cov, cholfact(cov))

type GMM{T,CM<:CovMat}
   n_dim::Integer               # dimensionality of data
   n_clust::Integer             # number of clusters
   weights::Array{T,1}          # 1d array of component weights
   means::Array{Array{T,1},1}   # 1d array of 1d mean arrays
   covs::Array{CM,1}            # 1d array of covariance matrices
   init::Bool                   # is the GMM initialized and ready to be fit?
   trained::Bool                # is the GMM fit to data?

   _wk::Array{T,1}              # reparameterized weights used in some training
end

type KMeans{T}
   n_dim::Integer               # dimensionality of data
   n_clust::Integer             # number of clusters
   means::Array{Array{T,1},1}   # 1d array of 1d mean arrays
   init::Bool                   # is the GMM initialized and ready to be fit?
   trained::Bool                # is the GMM fit to data?
end


## GMM constructors ##
function GMM{T}(
      X::Array{T,2};
      K::Int=3,
      cov_type::Symbol=:diag,
      mean_init_method::Symbol=:kmeans)

   n_ex, n_dim = size(X)
   (n_ex >= K && K > 0) || throw("Number of examples cannot be less than K")

   weights = ones(T, K)/T(K) # uniform

   # initialize means
   means = Array{Array{T, 1}, 1}(K)
   if mean_init_method == :zeros
      fill!(means, zeros(T, n_dim))
   
   elseif mean_init_method == :kmeans
      kmr = kmeans(X.', K; init=:kmpp)
      for ind in 1:K
         means[ind] = kmr.centers[:,ind]
      end

   elseif mean_init_method == :rand
      for ind in 1:K
         means[ind] = Array{T}(randn(n_dim))
      end

   else
      error("Unknown mean initialization method $(mean_init_method)")
   end

   # initialize covariance matrices to identity
   covs = Array{CovMat{T},1}(K)
   if cov_type == :diag
      for ind in 1:K
         covs[ind] = DiagCovMat(ones(T, n_dim))
      end

   elseif cov_type == :full
      for ind in 1:K
         covs[ind] = FullCovMat(eye(n_dim))
      end

   else
      error("Unknown covariance type $(cov_type).")
   end
   
   if cov_type == :diag
      return GMM{T,DiagCovMat{T}}(n_dim, K, weights, means, covs, true, false, Array{T}(K))
   elseif cov_type == :full
      return GMM{T,FullCovMat{T}}(n_dim, K, weights, means, covs, true, false, Array{T}(K))
   end
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


## make GMMs print a bit prettier ##
function print_cov_k{T,CM<:DiagCovMat}(io::IO, gmm::GMM{T,CM}, j::Integer; indent_level::Integer=0)
   pretty_print_vector(io, gmm.covs[j].diag, indent_level=indent_level)
end

function print_cov_k{T,CM<:FullCovMat}(io::IO, gmm::GMM{T,CM}, j::Integer; indent_level::Integer=0)
   pretty_print_matrix(io, gmm.covs[j].cov, indent_level=indent_level)
end

for (cm, cov_name) in ((:DiagCovMat, "diagonal"), (:FullCovMat, "full"))
   @eval begin
      function Base.show{T,CM<:$cm}(io::IO, gmm::GMM{T,CM})
         println(io, string("Gaussian mixture model in $(gmm.n_dim) ",
            "dimensions with $(gmm.n_clust) components with $($cov_name) ",
            "covariances:"))

         println(io, "  initialized: $(gmm.init)")
         println(io, "  trained:     $(gmm.trained)")
            
         #if gmm.init
         for k in 1:gmm.n_clust
            println(io, "  Component $(k):")
            println(io, "    weight:     $(gmm.weights[k])")
            println(io, "    mean:")
            pretty_print_vector(gmm.means[k], indent_level=6)
            println(io, "    cov ($($cov_name)):")
            print_cov_k(io, gmm, k, indent_level=6)
         end
         #end
      end
   end # @eval
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
 
