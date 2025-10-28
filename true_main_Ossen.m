function main_oseen_ls_dbc() % Oseen + Line Search + Corrected Direct BC (Clean)

clear all; close all;

fprintf('Solving Navier-Steady Stokes (Channel - Oseen + LS + Corrected Direct BC)...\n');

% --- 1. Parameters ---
mu = 1; % << SET VISCOSITY HERE (e.g., 1, 0.1, 0.01) >>
p_fem = 2; type = 'sym';
iter_max = 20; rtol = 1e-6; atol = 1e-8;
f1 = @(x,y) zeros(size(x)); f2 = @(x,y) zeros(size(x));

fprintf('*** Running simulation for mu = %.4f ***\n', mu);

% --- 2. Load Mesh & Generate P2 Info ---
fprintf('Loading mesh and parameters...\n');
loaded_data = load('domain_mesh.mat', 'p', 't', 'L', 'D', 'holes');
P = loaded_data.p; T = loaded_data.t; L = loaded_data.L; D = loaded_data.D; holes = loaded_data.holes;
Np = size(P, 1); num_holes = size(holes, 1);
[Pb, Tb] = FEmesh(P, T, p_fem); Npb = size(Pb, 1);
fprintf('Mesh Loaded: P1=%d, P2=%d\n', Np, Npb);

% --- 3. Define BC functions & Generate Dbc List ---
fprintf('Generating constraints...\n');
g1_inlet = @(x,y) atan(20*(D/2-abs(D/2-y))); g1_wall = @(x,y) zeros(size(x));
g2_all_dirichlet=@(x,y) zeros(size(x)); g3_pressure_pin=@(x,y) 0;
Dbc1 = index_val_Dirichlet_BC_channel(P, T, Pb, Tb, 0, D, holes, g1_inlet, g1_wall);
Dbc2_nodes = index_val_Dirichlet_BC_channel(P, T, Pb, Tb, 0, D, holes, g2_all_dirichlet, g2_all_dirichlet);
Dbc2 = []; if ~isempty(Dbc2_nodes); Dbc2 = [Dbc2_nodes(:,1) + Npb, Dbc2_nodes(:,2)]; end
[~, p_pin_idx] = min(sqrt(sum((P - [L/2,D/2]).^2, 2))); p_pin_dof = 2*Npb + p_pin_idx; Dbc3 = [p_pin_dof, g3_pressure_pin(P(p_pin_idx,1), P(p_pin_idx,2))];
Dbc = unique([Dbc1; Dbc2; Dbc3], 'rows');
fprintf('Identified %d unique constraints.\n', size(Dbc, 1));

% --- 4. Assemble Stokes Operator & Get Quadrature ---
fprintf('Assembling Stokes operator A0 & getting Gauss points...\n');
quad_order = 2*p_fem; % Use 5th order for potentially better accuracy in N(x) and An1
[gauss, weight] = gauss_integration(quad_order);

As_block = assemble_A_v(Pb, Tb, gauss, weight, p_fem, mu);
B_div = assemble_Bp_v(P, T, Pb, Tb, gauss, weight, p_fem, 1);
A0_saddle = assemble_spp(As_block, B_div, type); % Linear Stokes Operator
b0 = sparse(2*Npb+Np, 1); % Zero source term

% --- Define Residual Function Handle (Uses CORRECTED residual function) ---
residual_handle = @(xt) residual_navier_stokes(Pb, Tb, gauss, weight, p_fem, A0_saddle, xt, b0, Dbc);

% --- 5. Oseen Iteration with Line Search ---
fprintf('--- Starting Oseen Iteration with Line Search ---\n');
iter = 0;
x = zeros(2*Npb+Np, 1); % Initial guess x_0
res_iter = residual_handle(x); res_norm = norm(res_iter); res_0_norm = res_norm;
fprintf('Iter %d: Residual = %.4e\n', iter, res_norm);
tol = max(atol, res_0_norm * rtol); residual_history = [res_norm];

% Line Search Parameters
ls_options.alpha_init = 1.0; ls_options.beta = 0.5; ls_options.c = 1e-8;
ls_options.max_iter = 30; ls_options.min_alpha = 1e-8;

% Iteration Loop
while iter < iter_max && res_norm > tol
    iter = iter + 1;
    x_k = x;
    v0 = [x_k(1:Npb), x_k(Npb+1:2*Npb)];

    % Assemble Oseen matrix A_iter = A0 + An1(v0)
    An1_contribution = assemble_An1_v(P,T,Pb, Tb, gauss, weight, p_fem, v0);
    A_iter = A0_saddle;
    A_iter(1:(2*Npb), 1:(2*Npb)) = A_iter(1:(2*Npb), 1:(2*Npb)) + An1_contribution;
    b_iter = b0; % RHS for Oseen

    % Apply Corrected Direct BCs
    [A_bc, b_bc] = add_Dirichlet_BC(A_iter, b_iter, Dbc);

    % Solve linear system for potential step x_solved
    x_solved = A_bc \ b_bc;
    delta_x = x_solved - x_k; % Search direction

    % Perform Line Search using the TRUE non-linear residual
    res_norm_k = res_norm; % Residual norm before the step
    [x, alpha, success, res_norm] = perform_line_search(x_k, delta_x, res_norm_k, residual_handle, ls_options);

    %if ~success; warning('Iter %d: Line search failed. Stopping.', iter); break; end

    fprintf('Iter %d: Residual = %.4e (Accepted alpha=%.2e)\n', iter, res_norm, alpha);
    residual_history = [residual_history; res_norm];

    % No need to update tol inside loop if based on res_0_norm

end % End Oseen iteration loop

% --- Final Status ---
iterations_taken = iter; final_residual = res_norm;
if iter == iter_max && res_norm > tol; warning('Oseen+LS+DirectBC solver DID NOT converge for mu=%.4f.', mu); else; fprintf('Oseen+LS+DirectBC solver converged in %d iterations for mu=%.4f.\n', iter, mu); end

% --- 6. Post-Processing and Visualization ---
% (Visualization: Delaunay Vel, No Pressure, BC Plot)
fprintf('Extracting and plotting final results (Delaunay Velocity)...\n'); u_computed=x(1:Npb); v_computed=x(Npb+1:2*Npb); if issparse(u_computed); u_computed=full(u_computed); end; if issparse(v_computed); v_computed=full(v_computed); end; nodes_inside_holes_p2=false(Npb, 1); for i=1:num_holes; xc=holes(i,1); yc=holes(i,2); r=holes(i,3); nodes_inside_holes_p2=nodes_inside_holes_p2 | (dcircle(Pb, xc, yc, r) < -1e-8); end; u_computed(nodes_inside_holes_p2)=NaN; v_computed(nodes_inside_holes_p2)=NaN; fprintf('Generating Delaunay triangulation...\n'); try; tri_v=delaunay(Pb(:,1), Pb(:,2)); delaunay_ok=true; catch; warning('Delaunay failed for mu=%.4f', mu); delaunay_ok=false; end; if delaunay_ok; figure('Name', sprintf('U Vel (Oseen+LS+DBC mu=%.4f - Delaunay)', mu)); try; trisurf(tri_v, Pb(:,1), Pb(:,2), u_computed); title(sprintf('U (mu=%.4f)', mu)); xlabel('x'); ylabel('y'); colorbar; shading interp; view(2); axis equal; axis([0 L 0 D]); catch ME; warning('Plot U failed: %s',ME.message); end; figure('Name', sprintf('V Vel (Oseen+LS+DBC mu=%.4f - Delaunay)', mu)); try; trisurf(tri_v, Pb(:,1), Pb(:,2), v_computed); title(sprintf('V (mu=%.4f)', mu)); xlabel('x'); ylabel('y'); colorbar; shading interp; view(2); axis equal; axis([0 L 0 D]); catch ME; warning('Plot V failed: %s',ME.message); end; figure('Name', sprintf('Vel Mag (Oseen+LS+DBC mu=%.4f - Delaunay)', mu)); try; velocity_magnitude = sqrt(u_computed.^2 + v_computed.^2); trisurf(tri_v, Pb(:,1), Pb(:,2), velocity_magnitude); title(sprintf('|u| (mu=%.4f)', mu)); xlabel('x'); ylabel('y'); colorbar; shading interp; view(2); axis equal; axis([0 L 0 D]); catch ME; warning('Plot Mag failed: %s',ME.message); end; else; fprintf('Skipping Delaunay plots.\n'); end; figure('Name', sprintf('Streamlines (Oseen+LS+DBC mu=%.4f)', mu)); try; [X_reg, Y_reg] = meshgrid(linspace(0, L, 60), linspace(0, D, 40)); interp_u=scatteredInterpolant(Pb(:,1), Pb(:,2), u_computed,'linear','none'); interp_v=scatteredInterpolant(Pb(:,1), Pb(:,2), v_computed,'linear','none'); U_reg=interp_u(X_reg, Y_reg); V_reg=interp_v(X_reg, Y_reg); inside_holes_reg=false(size(X_reg)); for i=1:num_holes; points_reg=[X_reg(:), Y_reg(:)]; dist_vals_reg=dcircle(points_reg, holes(i,1), holes(i,2), holes(i,3)); inside_hole_i_reg=reshape(dist_vals_reg < -1e-8, size(X_reg)); inside_holes_reg=inside_holes_reg | inside_hole_i_reg; end; U_reg(inside_holes_reg)=NaN; V_reg(inside_holes_reg)=NaN; hold on; streamslice(X_reg, Y_reg, U_reg, V_reg, 1.5); plot([0 L L 0 0], [0 0 D D 0], 'k-', 'LineWidth', 1); for i = 1:num_holes; viscircles(holes(i,1:2), holes(i,3), 'Color', 'k', 'LineWidth', 1); end; hold off; title(sprintf('Streamlines (mu=%.4f)', mu)); xlabel('x'); ylabel('y'); axis equal; axis([0 L 0 D]); catch ME; warning('Plot Streamlines failed: %s', ME.message); end; fprintf('Identifying boundary nodes for plotting...\n'); try; Dbc1_plot=index_val_Dirichlet_BC_channel(P,T,Pb,Tb,0,D,holes,g1_inlet,g1_wall);Dbc2_plot_nodes=index_val_Dirichlet_BC_channel(P,T,Pb,Tb,0,D,holes,g2_all_dirichlet,g2_all_dirichlet);plot_nodes_u=[];if ~isempty(Dbc1_plot); plot_nodes_u=Dbc1_plot(:,1); end; plot_nodes_v=[]; if ~isempty(Dbc2_plot_nodes); plot_nodes_v=Dbc2_plot_nodes(:,1); end; plot_nodes_all_vel=unique([plot_nodes_u; plot_nodes_v]); plot_nodes_on_holes=[]; if ~isempty(plot_nodes_all_vel); coords_boundary=Pb(plot_nodes_all_vel,:); is_on_a_hole=false(length(plot_nodes_all_vel),1); hole_plot_tol=1e-2; for i=1:length(plot_nodes_all_vel); for h=1:num_holes; if abs(dcircle(coords_boundary(i,:), holes(h,1), holes(h,2), holes(h,3))) < hole_plot_tol; is_on_a_hole(i)=true; break; end; end; end; plot_nodes_on_holes=plot_nodes_all_vel(is_on_a_hole); plot_nodes_outer=setdiff(plot_nodes_all_vel, plot_nodes_on_holes); else; plot_nodes_on_holes=[]; plot_nodes_outer=[]; end; fprintf('Distinguished %d outer & %d hole nodes for plot.\n',length(plot_nodes_outer),length(plot_nodes_on_holes)); figure('Name',sprintf('Identified Boundary Nodes (Oseen+LS+DBC mu=%.4f)',mu)); triplot(T, P(:,1), P(:,2),'Color',[0.7 0.7 0.7]); hold on; if ~isempty(plot_nodes_outer); plot(Pb(plot_nodes_outer, 1), Pb(plot_nodes_outer, 2),'b.','MarkerSize',4,'DisplayName','Outer'); end; if ~isempty(plot_nodes_on_holes); plot(Pb(plot_nodes_on_holes, 1), Pb(plot_nodes_on_holes, 2),'ro','MarkerSize',5,'MarkerFaceColor','r','DisplayName','Holes'); end; title(sprintf('Identified P2 Boundary Nodes (Red=Holes), mu=%.4f',mu)); axis equal; axis([0 L 0 D]); xlabel('x'); ylabel('y'); legend show; for j=1:num_holes; viscircles(holes(j,1:2), holes(j,3),'Color','k','LineStyle','--'); end; hold off; catch ME_bcplot; warning('Could not plot boundary nodes: %s',ME_bcplot.message); end



% Plot residual history (consistent with Oseen version) ---  
if length(residual_history) > 1  
    fprintf('Plotting residual history...\n');  
    try  
        figure('Name', sprintf('Residual History [old] (Newton(Approx)+LS+DirectBC, mu=%.4e)', mu));  
        semilogy(0:length(residual_history) - 1, residual_history, '-o', 'LineWidth', 1.5, 'MarkerSize', 6);  
        xlabel('Iteration count');   
        ylabel('Residual norm [old] (log scale)');  
        title(sprintf('Convergence History mu=%.4e)', mu));  
        grid on;   
        set(gca, 'YScale', 'log');   
        hold on;  
        plot(get(gca, 'XLim'), [tol tol], 'r--', 'LineWidth', 1, 'DisplayName', sprintf('Tol=%.1e', tol));  
        legend('Residual [old]', 'Tolerance', 'Location', 'southwest');   
        hold off;  
    catch ME_resplot; warning('Error plotting residual history: %s', ME_resplot.message); end  
else;   
    fprintf('Insufficient residual history data (only %d points), skipping plot.\n', length(residual_history));   
end  

fprintf('\n--- Main function execution completed ---\n');  
fprintf('=============================================================================================\n');  

end % End of main function