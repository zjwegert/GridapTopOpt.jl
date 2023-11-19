struct MyIterator{A,B,C}
    cache :: Tuple{A,B,C}
end

function conv_criterion(m::MyIterator,item,state)
    it,J,C = item
    _,φ,dL = state
    return it>10
end

function Base.iterate(m::MyIterator)
    _,_,_ = m.cache
    J = 1; C = [1,1]
    return (0,J,C),(0,[],[])
end

function Base.iterate(m::MyIterator,state)
    (it,φ,dL) = state;
    a,b,c = m.cache

    J_new = a + rand()
    C_new = @. b + rand()

    new_item = (it+1,J_new,C_new)
    new_state = (it+1,φ,dL)
    
    if conv_criterion(m,new_item,new_state)
        return nothing
    else
        return new_item,new_state
    end
end

m = MyIterator((-1,-2,-3))
for (it,J_new,C_new) in m
    println("$it: $J_new | $C_new")
end