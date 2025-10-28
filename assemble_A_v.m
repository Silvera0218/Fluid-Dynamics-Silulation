function A_vv = assemble_A_v(Pb, Tb, gauss, weight, p, coef)
  A_u = assembleA_2D(Pb,Tb,gauss,weight,p,coef);
  A_vv = blkdiag(A_u, A_u);
end
