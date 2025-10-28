function A=assembleA_2D(Pb,Tb,gauss,weight,p,coef)
  Np=size(Pb,1);[Ne,Nlp]=size(Tb);Ng=size(gauss,1);
  dphix=basis_function(p,1,0,gauss);%referencebfderivativeforx
  dphiy=basis_function(p,0,1,gauss);%referencebfderivativefory
  A=sparse(Np,Np);
  for ne=1:Ne%loopall theelements
    x1=Pb(Tb(ne,1),1);y1=Pb(Tb(ne,1),2);
    x2=Pb(Tb(ne,2),1);y2=Pb(Tb(ne,2),2);
    x3=Pb(Tb(ne,3),1);y3=Pb(Tb(ne,3),2);
    detJ=(x2-x1)*(y3-y1)-(x3-x1)*(y2-y1);
    invJ=[(y3-y1),-(x3-x1);-(y2-y1),(x2-x1)]/detJ;
    S=zeros(Nlp,Nlp);%initializationlocalmatrix
    for i=1:Nlp%begin:localassembly
      for j=1:Nlp
         for k=1:Ng
           phi_i_dx=invJ(1,1)*dphix(i,k)+invJ(2,1)*dphiy(i,k);
           phi_j_dx=invJ(1,1)*dphix(j,k)+invJ(2,1)*dphiy(j,k);
           phi_i_dy=invJ(1,2)*dphix(i,k)+invJ(2,2)*dphiy(i,k);
           phi_j_dy=invJ(1,2)*dphix(j,k)+invJ(2,2)*dphiy(j,k);
           S(i,j)=S(i,j)+abs(detJ)*weight(k)*(phi_i_dx*phi_j_dx+phi_i_dy*phi_j_dy);
         end
       end
    end%end:localassembly
  A(Tb(ne,:),Tb(ne,:))=A(Tb(ne,:),Tb(ne,:))+S;%addlocalmatrixintoglobalmatrix
  end
  A=coef*A;
end

