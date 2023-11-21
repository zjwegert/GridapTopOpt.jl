# Material interpolation
Base.@kwdef struct SmoothErsatzMaterialInterpolation{M<:AbstractFloat}
  η::M # Smoothing radius
  ϵₘ::M = 10^-3 # Void material multiplier
  H = x -> H_η(x,η)
  DH = x -> DH_η(x,η)
  I = φ -> (1 - H(φ)) + ϵₘ*H(φ)
  ρ = φ -> 1 - H(φ)
end

# Heaviside function
function H_η(t,η)
  M = typeof(η*t)
  if t<-η
    return zero(M)
  elseif abs(t)<=η
    return 1/2*(1+t/η+1/pi*sin(pi*t/η))
  elseif t>η
    return one(M)
  end
end

function DH_η(t,η)
  M = typeof(η*t)
  if t<-η
    return zero(M)
  elseif abs(t)<=η
    return 1/2/η*(1+cos(pi*t/η))
  elseif t>η
    return zero(M)
  end
end
