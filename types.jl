
using Base.LinAlg: Cholesky
using Clustering: kmeans

## Define GMM types
abstract CovMat{T<:AbstractFloat}

type DiagCovMat{T<:AbstractFloat} <: CovMat{T}
   diag::Array{T,1} # main diagonal of covariance matrix
end

type FullCovMat{T<:AbstractFloat} <: CovMat{T}
   cov::Array{T,2} # full covariance matrix
   chol::Cholesky{T,Array{T,2}} # Cholesky factorization type of cov
end

function FullCovMat{T<:AbstractFloat}(cov::Array{T,2})
   return FullCovMat(cov, cholfact(cov))
end


type GMM{T<:AbstractFloat}
   weights::Array{T,1}          # 1d array of component weights
   means::Array{Array{T,1},1}   # 1d array of 1d mean arrays
   cov_type::Symbol             # the type of covariance matrix (:diag, :full)
   covs::Array{CovMat{T},1}     # 1d array of covariance matrices (of type CovMat)
   init::Bool                   # is the GMM initialized and ready to be fit?
   trained::Bool                # is the GMM fit to data?
end


# GMM constructors
function GMM{T<:AbstractFloat}(
      X::Array{T,2};
      k::Int=3,
      cov_type::Symbol=:diag,
      mean_init_method::Symbol=:kmeans)

   n_ex, n_dim = size(X)

   weights = ones(T, k)/T(k) # uniform

   # initialize means
   means = Array{Array{T, 1}, 1}(k)
   if mean_init_method == :zeros
      fill!(means, zeros(T, n_dim))
   
   elseif mean_init_method == :kmeans
      kmr = kmeans(X.', k; init=:kmpp)
      for ind in 1:k
         means[ind] = kmr.centers[:,ind]
      end

   elseif mean_init_method == :rand
      for ind in 1:k
         means[ind] = Array{T}(randn(n_dim))
      end

   else
      error("Unknown mean initialization method $(mean_init_method)")
   end

   # initialize covariance matrices to identity
   covs = Array{CovMat{T},1}(k)
   if cov_type == :diag
      #cm = DiagCovMat(ones(T, n_dim))
      for ind in 1:k
         covs[ind] = DiagCovMat(ones(T, n_dim))
      end

   elseif cov_type == :full
      #cm = FullCovMat(eye(n_dim))
      for ind in 1:k
         covs[ind] = FullCovMat(eye(n_dim))
      end

   else
      error("Unknown covariance type $(cov_type).")
   end

   return GMM(weights, means, cov_type, covs, true, false)
end


## make GMMs print a bit prettier ##
function Base.show{T<:AbstractFloat}(io::IO, gmm::GMM{T})
   function GMM_print_vector(io, indent_level, x)
      for val in x
         println(io, join([repeat(" ",indent_level), @sprintf "% 3.5f" val]))
      end
   end
   
   function GMM_print_matrix(io, indent_level, cov)
      for i in 1:size(cov,1)
         print(io, repeat(" ", indent_level))
         for val in cov[i,:]
            print(io, @sprintf "% 3.5f" val)
         end
         println(io,"")
      end
   end

   if gmm.init
      n_dim = size(gmm.means[1],1)
      k = size(gmm.weights,1)
      println(io, "Gaussian mixture model in $(n_dim) dimensions with $(k) components with $(gmm.cov_type) covariances:")

      println(io, "  initialized: $(gmm.init)")
      println(io, "  trained:     $(gmm.trained)")

      for j in 1:k
         println(io, "  Component $(j):")
         println(io, "    weight:     $(gmm.weights[j])")
         println(io, "    mean:")
         GMM_print_vector(io, 6, gmm.means[j])

         if gmm.cov_type == :diag
            println(io, "    cov (diag):")
            GMM_print_vector(io, 6, gmm.covs[j].diag)
         elseif gmm.cov_type == :full
            println(io, "    cov (full):")
            GMM_print_matrix(io, 6, gmm.covs[j].cov)
         end
      end

   else
      println(io, "Gaussian mixture model: uninitialized")
   end
end

