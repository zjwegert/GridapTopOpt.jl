function correct_ls!(x::Vector; tol = 1e-10)
  for i in eachindex(x)
    if iszero(abs(x[i]))
      x[i] = tol
        elseif abs(x[i]) < tol
      x[i] = tol*sign(x[i])
    end
  end
end

function correct_ls!(x::PVector; tol = 1e-10)
  map(local_views(x)) do x
    correct_ls!(x; tol)
  end
end

function correct_ls!(φh; tol = 1e-10)
  correct_ls!(get_free_dof_values(φh); tol)
end