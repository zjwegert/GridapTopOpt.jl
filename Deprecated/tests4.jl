
using Gridap
using Gridap.Arrays
using BenchmarkTools

struct Stencil{D,S,T} <: Map 
  Δ::NTuple{D,T}
  function Stencil(D::Integer,S::Tuple,Δ::Tuple)
    T = eltype(Δ)
    new{D,S,T}(Δ)
  end
end

@generated function get_stencil_indexes(::Stencil{2,S}, idx) where {S}
  ileft  = "(idx-1) % $(S[1]) == 0 ? -1 : idx - 1"
  iright = "(idx-1) % $(S[1]) == $(S[1]-1) ? -1 : idx + 1"
  idown  = "(idx-1) ÷ $(S[1]) == 0 ? -1 : idx - $(S[1])"
  iup    = "(idx-1) ÷ $(S[1]) == $(S[2]-1) ? - 1 : idx + $(S[1])"
  return Meta.parse("return ($ileft, $iright, $idown, $iup)")
end

@generated function compute_derivatives(s::Stencil{2,S},i,x) where S
  im = "i - 1"
  ip = "i + 1"
  jm = "i - $(S[1])"
  jp = "i + $(S[1])"

  dx⁻ = "(i-1) % $(S[1]) == 0 ? 0.0 : (x[i]-x[$im])/s.Δ[1]"
  dx⁺ = "(i-1) % $(S[1]) == $(S[1]-1) ? 0.0 : (x[$ip]-x[i])/s.Δ[1]"
  dy⁻ = "(i-1) ÷ $(S[1]) == 0 ? 0.0 : (x[i]-x[$jm])/s.Δ[2]"
  dy⁺ = "(i-1) ÷ $(S[1]) == $(S[2]-1) ? 0.0 : (x[$jp]-x[i])/s.Δ[2]"
  return Meta.parse("return ($dx⁻,$dx⁺,$dy⁻,$dy⁺)")
end

@generated function compute_derivatives(::Stencil{3,S},i,x) where S
  im = "i - 1"
  ip = "i + 1"
  jm = "i - $(S[1])"
  jp = "i + $(S[1])"
  km = "i - $(S[1])*$(S[2])"
  kp = "i + $(S[1])*$(S[2])"

  dx⁻ = "(i-1) % $(S[1]) == 0 ? 0.0 : (x[i]-x[$im])/s.Δ[1]"
  dx⁺ = "(i-1) % $(S[1]) == $(S[1]-1) ? 0.0 : (x[$ip]-x[i])/s.Δ[1]"
  dy⁻ = "((i-1) ÷ $(S[1]) % $(S[2])) == 0 ? 0.0 : (x[i]-x[$jm])/s.Δ[2]"
  dy⁺ = "((i-1) ÷ $(S[1]) % $(S[2])) == $(S[2]-1) ? 0.0 : (x[$jp]-x[i])/s.Δ[2]"
  dz⁻ = "(i-1) ÷ $(S[1]*S[2]) == 0 ? 0.0 : (x[i]-x[$km])/s.Δ[3]"
  dz⁺ = "(i-1) ÷ $(S[1]*S[2]) == $(S[3]-1) ? 0.0 : (x[$kp]-x[i])/s.Δ[3]"
  return Meta.parse("return ($dx⁻,$dx⁺,$dy⁻,$dy⁺,$dz⁻,$dz⁺)")
end

@generated function compute_derivatives(::Stencil{3,S},i,x,dx) where S
  im = "i - 1"
  ip = "i + 1"
  jm = "i - $(S[1])"
  jp = "i + $(S[1])"
  km = "i - $(S[1])*$(S[2])"
  kp = "i + $(S[1])*$(S[2])"

  body = ""
  body *= "dx[1] = (i-1) % $(S[1]) == 0 ? 0.0 : (x[i]-x[$im])/s.Δ[1]; "
  body *= "dx[2] = (i-1) % $(S[1]) == $(S[1]-1) ? 0.0 : (x[$ip]-x[i])/s.Δ[1]; "
  body *= "dx[3] = ((i-1) ÷ $(S[1]) % $(S[2])) == 0 ? 0.0 : (x[i]-x[$jm])/s.Δ[2]; "
  body *= "dx[4] = ((i-1) ÷ $(S[1]) % $(S[2])) == $(S[2]-1) ? 0.0 : (x[$jp]-x[i])/s.Δ[2]; "
  body *= "dx[5] = (i-1) ÷ $(S[1]*S[2]) == 0 ? 0.0 : (x[i]-x[$km])/s.Δ[3]; "
  body *= "dx[6] = (i-1) ÷ $(S[1]*S[2]) == $(S[3]-1) ? 0.0 : (x[$kp]-x[i])/s.Δ[3]; "
  return Meta.parse("begin $body return dx end")
end

macro index_to_tuple(idx, D1, D2, D3)
	return esc(:(
    ($idx - 1) % $D1 + 1, 
  (($idx - 1) ÷ $D1) % $D2 + 1, 
  ($idx - 1) ÷ ($D1 * $D2) + 1))
end

function reinit!(s::Stencil{2},φ_out,φ_in,S,Δt,caches)
  D⁻ˣ,D⁺ˣ,D⁻ʸ,D⁺ʸ,∇⁺,∇⁻ = caches
  for i in 1:length(φ_in)
    D⁻ˣ[i],D⁺ˣ[i],D⁻ʸ[i],D⁺ʸ[i] = compute_derivatives(s,i,φ_in)
  end

  ∇⁺ .= @. sqrt(max(D⁻ʸ,0)^2 + min(D⁺ʸ,0)^2 + max(D⁻ˣ,0)^2 + min(D⁺ˣ,0)^2)
  ∇⁻ .= @. sqrt(max(D⁺ʸ,0)^2 + min(D⁻ʸ,0)^2 + max(D⁺ˣ,0)^2 + min(D⁻ˣ,0)^2)
  φ_out .= @. φ_in - Δt*(max(S,0)*∇⁺ + min(S,0)*∇⁻ - S)
  return φ_out
end

function reinit!(s::Stencil{3},φ_out,φ_in,S,Δt,caches)
  D⁻ˣ,D⁺ˣ,D⁻ʸ,D⁺ʸ,D⁻ᶻ,D⁺ᶻ,∇⁺,∇⁻ = caches
  res = zeros(6)
  for i in 1:length(φ_in)
    compute_derivatives(s,i,φ_in,res)
    D⁻ˣ[i],D⁺ˣ[i],D⁻ʸ[i],D⁺ʸ[i],D⁻ᶻ[i],D⁺ᶻ[i] = res[1], res[2], res[3], res[4], res[5], res[6]
  end

  #∇⁺ .= @. sqrt(max(D⁻ʸ,0)^2 + min(D⁺ʸ,0)^2 + max(D⁻ˣ,0)^2 + min(D⁺ˣ,0)^2 + max(D⁻ᶻ,0)^2 + min(D⁺ᶻ,0)^2);
  #∇⁻ .= @. sqrt(max(D⁺ʸ,0)^2 + min(D⁻ʸ,0)^2 + max(D⁺ˣ,0)^2 + min(D⁻ˣ,0)^2 + max(D⁺ᶻ,0)^2 + min(D⁻ᶻ,0)^2);
  #φ_out .= @. φ_in - Δt*(max(S,0)*∇⁺ + min(S,0)*∇⁻ - S)
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

function reinit_step!(φ::T,φ_tmp::T,S::T,g_loc::NTuple{6,Bool},Δ::NTuple{3,M},Δt::M,caches) where {M,T<:Array{M,3}}
  (X⁻,X⁺,Y⁻,Y⁺,Z⁻,Z⁺) = g_loc
  D⁻ˣ,D⁺ˣ,D⁻ʸ,D⁺ʸ,D⁻ᶻ,D⁺ᶻ,∇⁺,∇⁻ = caches
  Δx,Δy,Δz = Δ 
  # Prepare shifted lsf
  circshift!(D⁺ʸ,φ,(0,-1)); circshift!(D⁻ʸ,φ,(0,1)); 
  circshift!(D⁺ˣ,φ,(-1,0)); circshift!(D⁻ˣ,φ,(1,0));
  circshift!(D⁺ᶻ,φ,(0,0,-1)); circshift!(D⁻ᶻ,φ,(0,0,1));
  # Forward (+) & Backward (-)
  D⁺ʸ .= (D⁺ʸ - φ)/Δy;
  D⁺ˣ .= (D⁺ˣ - φ)/Δx;
  D⁺ᶻ .= (D⁺ᶻ - φ)/Δz;
  D⁻ʸ .= (φ - D⁻ʸ)/Δy;
  D⁻ˣ .= (φ - D⁻ˣ)/Δx;
  D⁻ᶻ .= (φ - D⁻ᶻ)/Δz;
  # Check for boundaries with ghost nodes
  (~Y⁺) ? D⁺ʸ[:,end,:] .= zero(M) : 0;
  (~X⁺) ? D⁺ˣ[end,:,:] .= zero(M) : 0;
  (~Z⁺) ? D⁺ᶻ[:,:,end] .= zero(M) : 0;
  (~Y⁻) ? D⁻ʸ[:,1,:] .= zero(M) : 0;
  (~X⁻) ? D⁻ˣ[1,:,:] .= zero(M) : 0;
  (~Z⁻) ? D⁻ᶻ[:,:,1] .= zero(M) : 0;
  # Operators
  ∇⁺ .= @. sqrt(max(D⁻ʸ,0)^2 + min(D⁺ʸ,0)^2 + max(D⁻ˣ,0)^2 + min(D⁺ˣ,0)^2 + max(D⁻ᶻ,0)^2 + min(D⁺ᶻ,0)^2);
  ∇⁻ .= @. sqrt(max(D⁺ʸ,0)^2 + min(D⁻ʸ,0)^2 + max(D⁺ˣ,0)^2 + min(D⁻ˣ,0)^2 + max(D⁺ᶻ,0)^2 + min(D⁻ᶻ,0)^2);
  # Update
  φ_tmp .= @. φ - Δt*(max(S,0)*∇⁺ + min(S,0)*∇⁻ - S); 
  return nothing
end

# 2D 
sz = (1000,1012); n = prod(sz)
s = Stencil(2,sz,(0.5,0.5))

x = randn(n)
vel = ones(n); vel[[1,3,6,8,12,14]] .= 0;

y1 = zeros(n)
caches = zeros(n),zeros(n),zeros(n),zeros(n),zeros(n),zeros(n)
y1 = reinit!(s,y1,x,vel,0.1,caches)

y2 = reshape(copy(x),sz)
y_tmp = reshape(zeros(n),sz)
caches2 = zeros(sz),zeros(sz),zeros(sz),zeros(sz),zeros(sz),zeros(sz)
reinit_step!(y2,y_tmp,reshape(vel,sz),(false,false,false,false),(0.5,0.5),0.1,caches2)
#y_tmp = reshape(y_tmp,n)
#y_tmp ≈ y1

vel2 = reshape(vel,sz);

@benchmark reinit!(s,$y1,$x,$vel,0.1,$caches)
@benchmark reinit_step!($y2,$y_tmp,$vel2,(false,false,false,false),(0.5,0.5),0.1,$caches2)

# 3D 

sz = (104,108,10); n = prod(sz)
s = Stencil(3,sz,(0.5,0.5,0.5))

x = randn(n)
vel = ones(n); vel[[1,3,6,8,12,14]] .= 0;

y1 = zeros(n)
caches = zeros(n),zeros(n),zeros(n),zeros(n),zeros(n),zeros(n),zeros(n),zeros(n)
y1 = reinit!(s,y1,x,vel,0.1,caches)

y2 = reshape(copy(x),sz)
y_tmp = reshape(zeros(n),sz)
caches2 = zeros(sz),zeros(sz),zeros(sz),zeros(sz),zeros(sz),zeros(sz),zeros(sz),zeros(sz)
reinit_step!(y2,y_tmp,reshape(vel,sz),(false,false,false,false,false,false),(0.5,0.5,0.5),0.1,caches2)
#y_tmp = reshape(y_tmp,n)
#y_tmp ≈ y1

vel2 = reshape(vel,sz)

@benchmark reinit!($s,$y1,$x,$vel,0.1,$caches)
@benchmark reinit_step!($y2,$y_tmp,$vel2,(false,false,false,false,false,false),(0.5,0.5,0.5),0.1,$caches2)
