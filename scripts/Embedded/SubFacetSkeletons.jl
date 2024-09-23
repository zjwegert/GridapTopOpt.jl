
using GridapEmbedded
using GridapEmbedded.Interfaces

using Interfaces: SubFacetData, SubFacetTriangulation

struct SubFacetSkeletonData{Dp,T,Tn}
  subfacets::SubFacetData{Dp,T,Tn}
end

struct SubFacetSkeletonTriangulation{Dc,Dp} <: Triangulation{Dc,Dp}
  subfacets :: SubFacetData{Dp}
  subgrid :: UnstructuredGrid
  function SubFacetSkeletonTriangulation(
    subfacets::SubFacetData{Dp}
  ) where {Dp}
    Dc = Dp-2
    new{Dc,Dp}(subfacets,subgrid)
  end
end

function SkeletonTriangulation(trian::SubFacetTriangulation{Dc,Dp})
  subfacets = trian.subfacets
  subgrid = trian.subgrid
  return SubFacetSkeletonTriangulation(subfacets,subgrid)
end
