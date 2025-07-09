import GridapEmbedded.LevelSetCutters: update_trian!

# const MultiFieldSpaceTypes = Union{<:Gridap.MultiField.MultiFieldFESpace,<:GridapDistributed.DistributedMultiFieldFESpace}

function update_trian!(trian::DifferentiableTriangulation,space::MultiFieldSpaceTypes,φh)
  map((Ui,φi)->update_trian!(trian,Ui,φi),space,φh)
  return trian
end

function update_trian!(trian::DifferentiableTriangulation,::MultiFieldSpaceTypes,::Nothing)
  trian.cell_values = nothing
  return trian
end