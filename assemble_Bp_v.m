function B=assemble_Bp_v(P,T,Pb,Tb,gauss,weight,p,coef)
Npb=size(Pb,1);Np=size(P,1);
[Ne,Nlpj]=size(Tb);Nlpi=size(T,2);
Ng=size(gauss,1);
phi=basis_function(1,0,0,gauss);
dphix=basis_function(p,1,0,gauss); %reference bf derivative for x
dphiy=basis_function(p,0,1,gauss); %reference bf derivative for y
B1=sparse(Np,Npb);
B2=sparse(Np,Npb);
for ne=1:Ne %loop all the elements
    x1=Pb(Tb(ne,1),1);y1=Pb(Tb(ne,1),2);
    x2=Pb(Tb(ne,2),1);y2=Pb(Tb(ne,2),2);
    x3=Pb(Tb(ne,3),1);y3=Pb(Tb(ne,3),2);
    detJ=(x2-x1)*(y3-y1)-(x3-x1)*(y2-y1);
    invJ=[(y3-y1),-(x3-x1);-(y2-y1),(x2-x1)]/detJ;
    S1=zeros(Nlpi,Nlpj);%initialization local matrix 
    S2=zeros(Nlpi,Nlpj);%initialization local matrix
    for i=1:Nlpi %begin:local assembly
       for j=1:Nlpj
           for k=1:Ng
               phi_i=phi(i,k);
               phi_j_dx=invJ(1,1)*dphix(j,k)+invJ(2,1)*dphiy(j,k); %physical bf_j derivative for x
               phi_j_dy=invJ(1,2)*dphix(j,k)+invJ(2,2)*dphiy(j,k); %physical bf_j derivative for y
               S1(i,j)=S1(i,j)+abs(detJ)*weight(k)*(phi_i*phi_j_dx);
               S2(i,j)=S2(i,j)+abs(detJ)*weight(k)*(phi_i*phi_j_dy);
           end
       end
    end %end:local assembly
    B1(T(ne,:),Tb(ne,:))=B1(T(ne,:),Tb(ne,:))+S1;%add local matrix into global matrix
    B2(T(ne,:),Tb(ne,:))=B2(T(ne,:),Tb(ne,:))+S2;%add local matrix into global matrix
end
B=coef*[B1,B2];
end