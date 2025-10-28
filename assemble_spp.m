function A = assemble_spp(A_vv, B, type)

  Np = size(B, 1);        
  A = [ A_vv   -B'          ; 
        B      zeros(Np) ]; 

end