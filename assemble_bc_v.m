% (fv dot grad(fs),v)=(fv_1 fs_x + fv_2 fs_y, v)
% if fv(fs) is 1*2 (1*1) vector, fv(fs) is a constant vector
% if fv(fs) is Np*2 (Np*1) matrix, fv(fs) is a discretized solution 
function b=assemble_bc_v(Pb,Tb,gauss,weight,p,fv,fs)
Np=size(Pb,1);[Ne,Nlp]=size(Tb);Ng=size(gauss,1);
% vectorization process
x1=Pb(Tb(:,1),1);y1=Pb(Tb(:,1),2);
x2=Pb(Tb(:,2),1);y2=Pb(Tb(:,2),2);
x3=Pb(Tb(:,3),1);y3=Pb(Tb(:,3),2);
detJ=(x2-x1).*(y3-y1)-(x3-x1).*(y2-y1);
invJ11=(y3-y1)./detJ;invJ12=-(x3-x1)./detJ;
invJ21=-(y2-y1)./detJ;invJ22=(x2-x1)./detJ;
% basis functions
phi=basis_function(p,0,0,gauss);
dphix=basis_function(p,1,0,gauss);
dphiy=basis_function(p,0,1,gauss);
% fv(gauss)
type_v=size(fv,1);
if type_v==1
    fv=ones(Np,1)*fv;
end
fv_1=zeros(Ne,Ng); fv_2=zeros(Ne,Ng);
for k=1:Ng
    for i=1:Nlp
        fv_1(:,k)=fv_1(:,k)+fv(Tb(:,i),1)*phi(i,k);
        fv_2(:,k)=fv_2(:,k)+fv(Tb(:,i),2)*phi(i,k);
    end
end
type_s=size(fs,1);
if type_s==1
    fs=ones(Np,1)*fs;
end
fs_x=zeros(Ne,Ng);fs_y=zeros(Ne,Ng);
for k=1:Ng
    for i=1:Nlp
        phi_i_dx=invJ11*dphix(i,k)+invJ21*dphiy(i,k);
        phi_i_dy=invJ12*dphix(i,k)+invJ22*dphiy(i,k);
        fs_x(:,k)=fs_x(:,k)+fs(Tb(:,i)).*phi_i_dx;
        fs_y(:,k)=fs_y(:,k)+fs(Tb(:,i)).*phi_i_dy;
    end
end
f=fv_1.*fs_x+fv_2.*fs_y;
clear fv_1 fv_2 fs_x fs_y
% preallocate spaces
ii=zeros(Nlp*Ne,1);ss=zeros(Nlp*Ne,1);
index=0;
for i=1:Nlp
    S=zeros(Ne,1);
    for k=1:Ng       
        S=S+abs(detJ).*weight(k).*phi(i,k).*f(:,k);
    end
    ii(index+1:index+Ne)=Tb(:,i);
    ss(index+1:index+Ne,:)=S;
    index=index+Ne;
end
b=accumarray(ii,ss,[Np 1]); 
end