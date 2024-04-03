

function evaluate_cache(a::AbstractVector{<:IntegrandOperator},uh)
  return evaluate_cache(first(a),uh)
end

function Arrays.evaluate!(cache,a::AbstractVector{<:IntegrandOperator},uh)
  map(enumerate(a)) do (i,ai)
    evaluate!(cache,ai,uh;updated=(i!=1))
  end
end

function Arrays.evaluate(a::AbstractVector{<:IntegrandOperator},uh)
  cache = evaluate_cache(a,uh)
  return evaluate!(cache,a,uh)
end

function gradient_cache(a::AbstractVector{<:IntegrandOperator},uh,K)
  return gradient_cache(first(a),uh,K)
end

function gradient!(cache,a::AbstractVector{<:IntegrandOperator},uh,K)
  map(enumerate(a)) do (i,ai)
    gradient!(cache,ai,uh,K;updated=(i!=1))
  end
end

function Gridap.gradient(a::AbstractVector{<:IntegrandOperator},uh,K)
  cache = gradient_cache(a,uh,K)
  return gradient!(cache,a,uh,K)
end

function evaluate_and_gradient_cache(a::AbstractVector{<:IntegrandOperator},uh,K)
  return evaluate_and_gradient_cache(first(a),uh,K)
end

function evaluate_and_gradient!(cache,a::AbstractVector{<:IntegrandOperator},uh,K)
  map(enumerate(a)) do (i,ai)
    evaluate_and_gradient!(cache,ai,uh,K;updated=(i!=1))
  end
end

function evaluate_and_gradient(a::AbstractVector{<:IntegrandOperator},uh,K)
  cache = evaluate_and_gradient_cache(a,uh,K)
  return evaluate_and_gradient!(cache,a,uh,K)
end
