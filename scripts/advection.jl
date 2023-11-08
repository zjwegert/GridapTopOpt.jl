
# This file shows how to modify the advection scheme

using GridapTopOp

struct MyStencil <: GridapTopOp.AdvectionStencil end

function GridapTopOp.allocate_caches(::MyStencil,φ,vel)
  # Do stuff here...
end

function GridapTopOp.reinit!(::MyStencil,φ_new,φ_old,vel,Δt,Δx,caches)
  # Do stuff here...
end

function GridapTopOp.advect!(::MyStencil,φ,vel,Δt,Δx,caches)
  # Do stuff here...
end

function GridapTopOp.compute_Δt(s::MyStencil{D,T},φ,vel) where {D,T}
  # Do stuff here...
end

########################################

# Basically same code as the standard one, but using our new stencil
function main()

end

