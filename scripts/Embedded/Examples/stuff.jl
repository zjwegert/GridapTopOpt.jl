function finished(h)
  _A = []
  _B = []
  for it = 10:size(hist,1)
    Li, Ci = h[1,it], h[3,it]
    L_prev = h[1,it-5:it-1]
    push!(_A,map(L -> abs(Li - L)/abs(Li), L_prev))
    push!(_B,abs(Ci))
  end
  return _A,_B
end