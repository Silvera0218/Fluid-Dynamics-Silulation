function[Pb,Tb]=FEmesh(P,T,p)
  if p==1
    Pb=P;Tb=T;
    figure; triplot(T,P(:,1),P(:,2));axis equal;
  elseif p==2
    Np=size(P,1);Ne=size(T,1);
    Pnew=[];%intializenewpoints
    Tnew=zeros(Ne,3);%intializeglobalmap([456])
    for i=1:Ne
      P1=P(T(i,1),:);P2=P(T(i,2),:);P3=P(T(i,3),:);
      P4=(P1+P2)./2;P5=(P2+P3)./2;P6=(P1+P3)./2;
      if i==1
        Pnew=[P4;P5;P6];
        Tnew(i,:)=[1 2 3];
      else
        [~,id]=ismember(P4,Pnew,'rows');%P4
        if id==0
          Pnew=[Pnew;P4];Tnew(i,1)=size(Pnew,1);
        else
          Tnew(i,1)=id;
        end
        [~,id]=ismember(P5,Pnew,'rows');%P5
        if id==0
          Pnew=[Pnew;P5];Tnew(i,2)=size(Pnew,1);
        else
          Tnew(i,2)=id;
        end
        [~,id]=ismember(P6,Pnew,'rows');%P6
        if id==0
          Pnew=[Pnew;P6];Tnew(i,3)=size(Pnew,1);
        else
        Tnew(i,3)=id;
        end
      end
    end
    Pb=[P;Pnew];Tb=[T,Np+Tnew];
    figure;
    triplot(T,P(:,1),P(:,2));axis equal;hold on;
    triplot(Tnew,Pnew(:,1),Pnew(:,2),'m');hold off;
    title('FE mesh');
  end
end