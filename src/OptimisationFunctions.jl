# We may not need this kind of thing? Depends how we want to specify functions and their
# shape derivatives.

# # We expect f and v_f are functions that take ... 
# struct OptimisationFunction
#     name::String
#     f           # Constraint or objective function 
#     v_f         # Shape sensitivity of f
#     v_f_space
#     v_f_cache
#     function OptimisationFunction(name,f,v_f,v_f_space)
#         v_f_cache = FEFunction(...)
#         new(name,f,v_f,v_f_space,v_f_cache)
#     end
# end

# function evaluate(opt_fn::OptimisationFunction,φh,data)
#     return opt_fn.f(φh,data)
# end

# function evaluate_shape_sens(opt_fn::OptimisationFunction,φh,data)
#     return opt_fn.v_f(φh,data)
# end

# function *(a::Number,b::OptimisationFunction) 
#     v_f = (φh,data) -> FEFunction(b.v_f_space,a*get_free_dof_values(b.v_f(φh,data)))
#     OptimisationFunction("$a*"*b.name,(φh,data)->a*b.f(φh,data),v_f,b.v_f_space);
# end
# *(b::OptimisationFunction,a::Number) = a*b

# function /(a::OptimisationFunction,b::Number) 
#     v_f = (φh,data) -> FEFunction(a.v_f_space,get_free_dof_values(a.v_f(φh,data))/a)
#     OptimisationFunction(a.name*"/$b",(φh,data)->b.f(φh,data)/a,v_f,a.v_f_space);
# end
# function +(a::OptimisationFunction,b::Number)
#     OptimisationFunction(a.name*"+$b",(φh,data)->b.f(φh,data)+b,a.v_f,a.v_f_space);

#     v_f = (x,y) -> FEFunction(get_fe_space(first(first(y.hom_data.v_Cᴴ))),
#         a.v_f(x,y).free_values)
#     OptimisationFunction(a.name*" + "*string(b),(x,y)->a.f(x,y)+b,v_f);
# end
# function -(a::OptimisationFunction,b::Number) 
#     v_f = (x,y) -> FEFunction(get_fe_space(first(first(y.hom_data.v_Cᴴ))),
#         a.v_f(x,y).free_values)
#     OptimisationFunction(a.name*" - "*string(b),(x,y)->a.f(x,y)-b,v_f);
# end
# function +(b::Number,a::OptimisationFunction) 
#     v_f = (x,y) -> FEFunction(get_fe_space(first(first(y.hom_data.v_Cᴴ))),
#         a.v_f(x,y).free_values)
#     OptimisationFunction(a.name*" + "*string(b),(x,y)->a.f(x,y)+b,v_f);
# end
# function -(b::Number,a::OptimisationFunction) 
#     v_f = (x,y) -> FEFunction(get_fe_space(first(first(y.hom_data.v_Cᴴ))),
#         -a.v_f(x,y).free_values)
#     OptimisationFunction(a.name*" - "*string(b),(x,y)->b-a.f(x,y),v_f);
# end
# function +(a::OptimisationFunction,b::OptimisationFunction) 
#     v_f = (x,y) -> FEFunction(get_fe_space(first(first(y.hom_data.v_Cᴴ))),
#         a.v_f(x,y).free_values+b.v_f(x,y).free_values)
#     OptimisationFunction(a.name*" + "*b.name,(x,y)->a.f(x,y)+b.f(x,y),v_f);
# end
# function -(a::OptimisationFunction,b::OptimisationFunction) 
#     v_f = (x,y) -> FEFunction(get_fe_space(first(first(y.hom_data.v_Cᴴ))),
#         a.v_f(x,y).free_values-b.v_f(x,y).free_values)
#     OptimisationFunction(a.name*" - "*b.name,(x,y)->a.f(x,y)-b.f(x,y),v_f);
# end
# function -(a::OptimisationFunction) 
#     v_f = (x,y) -> FEFunction(get_fe_space(first(first(y.hom_data.v_Cᴴ))),
#         -a.v_f(x,y).free_values)
#     OptimisationFunction("-"*a.name,(x,y)->-a.f(x,y),v_f);
# end
