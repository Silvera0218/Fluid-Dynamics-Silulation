function [A,b]=add_Dirichlet_BC(A,b,Dbc)
  [index,val]=deal(Dbc(:,1),Dbc(:,2));
  Ni=length(index);
  b=b-A(:,index)*val;
  b(index)=val;
  A(:,index)=0;
  A(index,:)=0;
  for i=1:Ni
    A(index(i),index(i))=1;
  end
end