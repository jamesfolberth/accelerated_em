
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

