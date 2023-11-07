
using Gridap
using Gridap.Arrays

struct Stencil{D,S,T} <: Map 
  Δx::NTuple{D,T}
end

@generated function get_stencil_indexes(::Stencil{2,S}, idx) where {S}
  ileft  = "(idx-1) % $(S[1]) == 0 ? -1 : idx - 1"
  iright = "(idx-1) % $(S[1]) == $(S[1]-1) ? -1 : idx + 1"
  idown  = "(idx-1) ÷ $(S[1]) == 0 ? -1 : idx - $(S[1])"
  iup    = "(idx-1) ÷ $(S[1]) == $(S[2]-1) ? - 1 : idx + $(S[1])"
  return Meta.parse("return ($ileft, $iright, $idown, $iup)")
end

@generated function compute_derivatives(::Stencil{2,S},i,x,dx,dy) where S
  il = "i - 1"
  ir = "i + 1"
  id = "i - $(S[1])"
  iu = "i + $(S[1])"

  dx⁻ = "(i-1) % $(S[1]) == 0 ? 0.0 : (x[i]-x[$il])/dx"
  dx⁺ = "(i-1) % $(S[1]) == $(S[1]-1) ? 0.0 : (x[$ir]-x[i])/dx"
  dy⁻ = "(i-1) ÷ $(S[1]) == 0 ? 0.0 : (x[i]-x[$id])/dy"
  dy⁺ = "(i-1) ÷ $(S[1]) == $(S[2]-1) ? 0.0 : (x[$iu]-x[i])/dy"
  return Meta.parse("return ($dx⁻,$dx⁺,$dy⁻,$dy⁺)")
end

function reinit!(s::Stencil{2},φ_out,φ_in,S,Δ,Δt,caches)
  D⁻ˣ,D⁺ˣ,D⁻ʸ,D⁺ʸ,∇⁺,∇⁻ = caches
  dx,dy = Δ
  for i in 1:length(φ_in)
    D⁻ˣ[i],D⁺ˣ[i],D⁻ʸ[i],D⁺ʸ[i] = compute_derivatives(s,i,φ_in,dx,dy)
  end

  ∇⁺ .= @. sqrt(max(D⁻ʸ,0)^2 + min(D⁺ʸ,0)^2 + max(D⁻ˣ,0)^2 + min(D⁺ˣ,0)^2)
  ∇⁻ .= @. sqrt(max(D⁺ʸ,0)^2 + min(D⁻ʸ,0)^2 + max(D⁺ˣ,0)^2 + min(D⁻ˣ,0)^2)
  φ_out .= @. φ_in - Δt*(max(S,0)*∇⁺ + min(S,0)*∇⁻ - S)
  return φ_out
end

function reinit_step!(φ::T,φ_tmp::T,S::T,g_loc::NTuple{4,Bool},Δ::NTuple{2,M},Δt::M,caches) where {M,T<:Array{M,2}}
  (X⁻,X⁺,Y⁻,Y⁺) = g_loc
  D⁻ˣ,D⁺ˣ,D⁻ʸ,D⁺ʸ,∇⁺,∇⁻ = caches
  Δx,Δy = Δ 
  # Prepare shifted lsf
  circshift!(D⁺ʸ,φ,(0,-1)); circshift!(D⁻ʸ,φ,(0,1)); 
  circshift!(D⁺ˣ,φ,(-1,0)); circshift!(D⁻ˣ,φ,(1,0)); 
  # Forward (+) & Backward (-)
  D⁺ʸ .= @. (D⁺ʸ - φ)/Δy;
  D⁺ˣ .= @. (D⁺ˣ - φ)/Δx;
  D⁻ʸ .= @. (φ - D⁻ʸ)/Δy;
  D⁻ˣ .= @. (φ - D⁻ˣ)/Δx;
  # Check for boundaries with ghost nodes
  (~Y⁺) ? D⁺ʸ[:,end] .= zero(M) : 0;
  (~X⁺) ? D⁺ˣ[end,:] .= zero(M) : 0;
  (~Y⁻) ? D⁻ʸ[:,1] .= zero(M) : 0;
  (~X⁻) ? D⁻ˣ[1,:] .= zero(M) : 0;
  # Operators
  ∇⁺ .= @. sqrt(max(D⁻ʸ,0)^2 + min(D⁺ʸ,0)^2 + max(D⁻ˣ,0)^2 + min(D⁺ˣ,0)^2);
  ∇⁻ .= @. sqrt(max(D⁺ʸ,0)^2 + min(D⁻ʸ,0)^2 + max(D⁺ˣ,0)^2 + min(D⁻ˣ,0)^2);
  # Update
  φ_tmp .= @. φ - Δt*(max(S,0)*∇⁺ + min(S,0)*∇⁻ - S); 
  return nothing
end

sz = (1000,1012); n = prod(sz)
s = Stencil{2,sz}()

x = randn(n)
vel = ones(n); vel[[1,3,6,8,12,14]] .= 0;

y1 = zeros(n)
caches = zeros(n),zeros(n),zeros(n),zeros(n),zeros(n),zeros(n)
y1 = reinit!(s,y1,x,vel,(0.5,0.5),0.1,caches)

y2 = reshape(copy(x),sz)
y_tmp = reshape(zeros(n),sz)
caches2 = zeros(sz),zeros(sz),zeros(sz),zeros(sz),zeros(sz),zeros(sz)
reinit_step!(y2,y_tmp,reshape(vel,sz),(false,false,false,false),(0.5,0.5),0.1,caches2)
#y_tmp = reshape(y_tmp,n)
#y_tmp ≈ y1

vel2 = reshape(vel,sz)

using BenchmarkTools

@benchmark reinit!(s,$y1,$x,$vel,(0.5,0.5),0.1,$caches)
@benchmark reinit_step!($y2,$y_tmp,$vel2,(false,false,false,false),(0.5,0.5),0.1,$caches2)

