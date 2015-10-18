"""
Top-level module for accelerated KMeans and GMMs.

This module has a few types of GMM and will eventually have a few different
algorithms for training them.  This is totally experimental!

There are other GMM modules in Julia:
   https://github.com/davidavdav/GaussianMixtures.jl
   https://github.com/lindahua/MixtureModels.jl


This module has an implementation of k-means with a few different training methods.
This is totally experimental!

There are other k-means implementations in Julia:
https://github.com/JuliaStats/Clustering.jl
"""

module EMAccel

using PyCall
using PyPlot # need contour, show, ax
@pyimport matplotlib.patches as mpatches # for plotting KMeans means

# GMM
export GMM
export em!, compute_ll
export gd!
export data_sanity, score_samples, plot_gmm_contours

include("GMMAccel/types.jl")
include("GMMAccel/em.jl")
include("GMMAccel/gd.jl")
include("GMMAccel/utils.jl")


# KMeans
export KMeans
export hard_em!, em!, compute_ll
export gd!, nest2!
export data_sanity, hard_classify, soft_classify, plot_means

include("KMeansAccel/types.jl")
include("KMeansAccel/em.jl")
include("KMeansAccel/gd.jl")
include("KMeansAccel/utils.jl")

end
