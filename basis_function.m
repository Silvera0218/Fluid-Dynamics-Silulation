function phi=basis_function(p,ndx,ndy,gauss)
  n=(p+2)*(p+1)/2;m=size(gauss,1);
  phi=zeros(n,m);
  %P1element
  if p==1
   for j=1:m
    x=gauss(j,1);y=gauss(j,2);
    if ndx==0&&ndy==0 %basisfunctions
      phi(1,j)=-x-y+1; %point1:(0,0)
      phi(2,j)=x; %point2:(1,0)
      phi(3,j)=y; %point3:(0,1)
    end
    if ndx==1&&ndy==0 %firstderivativeforx
      phi(1,j)=-1;phi(2,j)=1;phi(3,j)=0;
    end
    if ndx==0&&ndy==1 %firstderivativefory
      phi(1,j)=-1;phi(2,j)=0;phi(3,j)=1;
    end
   end
  end
  %P2element
  if p==2
    for j=1:m
      x=gauss(j,1);y=gauss(j,2);
      if ndx==0 && ndy==0%basisfunctions
        phi(1,j)=2*x^2+2*y^2+4*x*y-3*y-3*x+1;%point1:(0,0)
        phi(2,j)=2*x^2-x;%point2:(1,0)
        phi(3,j)=2*y^2-y;%point3:(0,1)
        phi(4,j)=-4*x^2-4*x*y+4*x;%point4:(1/2,0)
        phi(5,j)=4*x*y;%point5:(1/2,1/2)
        phi(6,j)=-4*y^2-4*x*y+4*y;%point6:(0,1/2)
      end
      if ndx==1&&ndy==0%firstderivativeforx
        phi(1,j)=4*x+4*y-3;phi(2,j)=4*x-1;phi(3,j)=0;
        phi(4,j)=-8*x-4*y+4;phi(5,j)=4*y;phi(6,j)=-4*y;
      end
      if ndx==0&&ndy==1%firstderivativefory
        phi(1,j)=4*y+4*x-3;phi(2,j)=0;phi(3,j)=4*y-1;
        phi(4,j)=-4*x;phi(5,j)=4*x;phi(6,j)=-8*y-4*x+4;
      end
    end
  end
end