function res=residual_navier_stokes(Pb,Tb,gauss,weight,p,A0,x,b0,~) % Ignore Dbc
    Npb=size(Pb,1);Np=size(A0,1)-2*Npb;
    v=[x(1:Npb),x(Npb+1:2*Npb)]; % Current velocity v from input x

    % Calculate N(x) using the current velocity v
    bn1=assemble_bc_v(Pb,Tb,gauss,weight,p,v,v(:,1)); % Requires current v in BOTH velocity slots
    bn2=assemble_bc_v(Pb,Tb,gauss,weight,p,v,v(:,2)); % Requires current v in BOTH velocity slots
    N_x_vec=[bn1;bn2;sparse(Np,1)]; % This IS N(x)

    % Calculate residual F(x) = A0*x + N(x) - b0
    res = A0*x + N_x_vec - b0;
end