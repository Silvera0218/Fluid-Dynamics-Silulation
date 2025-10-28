function main_newton_ls_dbc_old_res_styled() % Newton + LS + Direct BC + Old Residual (Styled like Oseen)  

clear all; close all; clc;  

fprintf('Solving steady-state Navier-Stokes (Channel Flow - Newton(approx) + LS + Direct BC [using old residual function, Oseen style])...\n');  
fprintf('=============================================================================================\n');  

% --- 1. Parameter settings ---  
mu = 0.1; % Viscosity coefficient  
p_fem = 2; type = 'sym';  
iter_max = 20; % (Adjustable)  
rtol = 1e-6;  
atol = 1e-8;  
f1 = @(x,y) zeros(size(x)); % Source term (zero)  
f2 = @(x,y) zeros(size(x));  

% --- 2. Load mesh & generate P2 information ---  
fprintf('Loading mesh and parameters...\n');  
mesh_file = 'domain_mesh.mat';  
if ~exist(mesh_file, 'file'); error('Mesh file %s not found!', mesh_file); end  
try  
    loaded_data = load(mesh_file, 'p', 't', 'L', 'D', 'holes');   
    P = loaded_data.p; T = loaded_data.t; L = loaded_data.L;   
    D = loaded_data.D; holes = loaded_data.holes;   
catch ME; error('Cannot load mesh file %s: %s', mesh_file, ME.message); end  
Np = size(P, 1);   
[Pb, Tb] = FEmesh(P, T, p_fem);   
Npb = size(Pb, 1);   
num_holes = size(holes, 1);  
fprintf('Mesh loaded: P1=%d nodes, P2=%d nodes\n', Np, Npb);  

% --- 3. Define BC functions & generate Dbc list ---  
fprintf('Generating constraint list Dbc...\n');  
g1_inlet = @(x,y) atan(20*(D/2-abs(D/2-y)));   
g1_wall = @(x,y) zeros(size(x));  
g2_all_dirichlet = @(x,y) zeros(size(x));   
g3_pressure_pin = @(x,y) 0;  
try  
    Dbc1 = index_val_Dirichlet_BC_channel(P, T, Pb, Tb, 0, D, holes, g1_inlet, g1_wall);  
    Dbc2_nodes = index_val_Dirichlet_BC_channel(P, T, Pb, Tb, 0, D, holes, g2_all_dirichlet, g2_all_dirichlet);   
    Dbc2 = [];   
    if ~isempty(Dbc2_nodes); Dbc2 = [Dbc2_nodes(:,1) + Npb, Dbc2_nodes(:,2)]; end  
    [~, p_pin_idx] = min(sqrt(sum((P - [L/2, D/2]).^2, 2)));   
    p_pin_dof = 2*Npb + p_pin_idx;   
    Dbc3 = [p_pin_dof, g3_pressure_pin(P(p_pin_idx,1), P(p_pin_idx,2))];  
    Dbc = unique([Dbc1; Dbc2; Dbc3], 'rows');  
    fprintf('Identified %d unique constraints.\n', size(Dbc, 1));  
catch ME_dbc; error('Error generating Dbc list: %s.', ME_dbc.message); end  

% --- 4. Assemble Stokes operator & obtain Gaussian points ---  
fprintf('Assembling Stokes operator A0 & obtaining Gaussian points...\n');  
quad_order = 2*p_fem;  
[gauss_bary, weight] = gauss_integration(quad_order); % Get barycentric coordinates  
fprintf('Obtained %d Gaussian points (barycentric coordinates).\n', size(gauss_bary,1));  
tic_const = tic;  
As_block = assemble_A_v(Pb, Tb, gauss_bary, weight, p_fem, mu);      % Internal conversion  
B_div = assemble_Bp_v(P, T, Pb, Tb, gauss_bary, weight, p_fem, 1); % Internal conversion  
A0_saddle = assemble_spp(As_block, B_div, type); % Linear Stokes operator  
fprintf('Constant matrix assembly completed (time taken %.2f seconds).\n', toc(tic_const));  
b0 = sparse(2*Npb + Np, 1); % Zero source term  

% --- 5. Define residual function handle (using old residual function) ---  
fprintf('Defining residual function handle (using old residual_navier_stokes)...\n');  
residual_handle = @(xt) residual_navier_stokes(Pb, Tb, gauss_bary, weight, p_fem, A0_saddle, xt, b0, Dbc);  

% --- 6. Newton (approx) iteration ---  
fprintf('--- Starting Newton (approx) iteration (line search, using old residual function for assessment) ---\n');  
iter = 0;   
x = zeros(2*Npb + Np, 1);   
final_residual = NaN;   
iterations_taken = 0;   
success = true;  

% Compute initial residual  
fprintf('Calculating initial residual (using old residual function)...\n');  
try   
    res_iter = residual_handle(x);   
    res_norm = norm(res_iter);  
catch ME_res_init; error('Error calculating initial residual: %s.', ME_res_init.message); end  
if ~isfinite(res_norm); error('Initial residual is NaN/Inf.'); end  
res_0_norm = res_norm;   
fprintf('Iter %d: Initial residual norm [old] = %.4e\n', iter, res_norm);  
tol = max(atol, res_0_norm * rtol);   
fprintf('Convergence tolerance tol = %.4e\n', tol);  
residual_history = [res_norm];  

if res_norm <= tol  
    fprintf('******************* Initial convergence *******************\n');  
    success = true;   
    iterations_taken = 0;   
    final_residual = res_norm;  
else  
    % --- Line search parameters ---  
    ls_options.alpha_init = 1.0;   
    ls_options.beta = 0.5;   
    ls_options.c = 1e-4;  
    ls_options.max_iter = 15;    
    ls_options.min_alpha = 1e-7;  
    fprintf('Line search parameters: alpha_init=%.1f, beta=%.1f, c=%.1e, max_iter=%d, min_alpha=%.1e\n', ...  
            ls_options.alpha_init, ls_options.beta, ls_options.c, ls_options.max_iter, ls_options.min_alpha);  

    % --- Newton (approx) iteration loop ---  
    tic_iteration = tic;  
    while iter < iter_max && res_norm > tol  
        iter = iter + 1;  
        x_k = x; % Current solution u_{k-1}  
        v0 = [x_k(1:Npb), x_k(Npb+1:2*Npb)]; % v0 = u_{k-1}  

        fprintf('--- Iter %d (current residual [old]: %.4e) ---\n', iter, res_norm);  

        % 1. Assemble An1(v0)  
        fprintf('  1. Assembling An1(v0)...\n');   
        tic_an1 = tic;  
        try   
            An1 = assemble_An1_v(P, T, Pb, Tb, gauss_bary, weight, p_fem, v0);  
        catch ME_an1; warning('Iter %d: Error assembling An1: %s', iter, ME_an1.message); success = false; break; end  
        fprintf('     An1 assembly completed (%.2f seconds).\n', toc(tic_an1));  

        % 2. Assemble An2(v0) (approximate Newton)  
        fprintf('  2. Assembling An2(v0) (approximate)...\n');   
        tic_an2 = tic;  
        try   
            An2 = assemble_An2_v(P, T, Pb, Tb, gauss_bary, weight, p_fem, v0); % Use approximate version  
        catch ME_an2; warning('Iter %d: Error assembling An2: %s', iter, ME_an2.message); success = false; break; end  
        fprintf('     An2 (approx) assembly completed (%.2f seconds).\n', toc(tic_an2));  

        % 3. Calculate Fn(v0) vector = N(v0)  
        fprintf('  3. Calculating Fn(v0) vector...\n');   
        tic_fn = tic;  
        try  
            Fn1 = assemble_bc_v(Pb, Tb, gauss_bary, weight, p_fem, v0, v0(:,1));  
            Fn2 = assemble_bc_v(Pb, Tb, gauss_bary, weight, p_fem, v0, v0(:,2));  
            Fn_vec = [Fn1; Fn2; sparse(Np,1)]; % Fn vector  
        catch ME_fn; warning('Iter %d: Error calculating Fn: %s', iter, ME_fn.message); success = false; break; end  
        fprintf('     Fn vector calculation completed (%.2f seconds).\n', toc(tic_fn));  

        % 4. Construct Newton (approx) system matrix A_iter and RHS b_iter  
        fprintf('  4. Constructing Newton (approx) system...\n');  
        A_iter = A0_saddle; % Start with Stokes Operator  
        % Add An1 and An2 to the velocity-velocity block  
        A_iter(1:(2*Npb), 1:(2*Npb)) = A_iter(1:(2*Npb), 1:(2*Npb)) + An1 + An2; % Add both An1 and An2  
        b_iter = b0 + Fn_vec; % RHS = F + Fn (according to the formula, F=b0=0)  

        % 5. Apply boundary conditions using direct elimination method  
        fprintf('  5. Applying boundary conditions using direct elimination method...\n');  
        try   
            [A_bc, b_bc] = add_Dirichlet_BC(A_iter, b_iter, Dbc); % <--- Use direct method  
        catch ME_bc; warning('Iter %d: Error applying boundary conditions (direct method): %s', iter, ME_bc.message); success = false; break; end  

        % 6. Solve linear system to get the next iteration solution x_solved  
        fprintf('  6. Solving Newton (approx) linear system...\n');   
        tic_solve = tic;  
        try  
            x_solved = A_bc \ b_bc; % Direct solve for x_k  
            if any(isnan(x_solved)) || any(isinf(x_solved)); warning('Iter %d: Linear system solution contains NaN/Inf!', iter); success = false; break; end  
        catch ME_solve; warning('Iter %d: Error solving linear system: %s', iter, ME_solve.message); success = false; break; end  
        fprintf('     Linear system solve completed (%.2f seconds).\n', toc(tic_solve));  

        % 7. Calculate the "direction" toward the new solution for line search  
        delta_x = x_solved - x_k;  
        norm_delta_x = norm(delta_x);  
        fprintf('  7. Calculating search direction (x_solved - x_k), ||delta_x|| = %.4e\n', norm_delta_x);  
        if norm_delta_x < 1e-14; fprintf('     Search direction too small, iteration terminating.\n'); break; end  

        % 8. Execute line search (using old residual function for assessment)  
        res_norm_k = res_norm;  
        fprintf('  8. Executing line search (target: reduce residual norm [old] %.4e)...\n', res_norm_k);   
        tic_ls = tic;  
        try  
            [x, alpha, success_ls, res_norm] = perform_line_search(x_k, delta_x, res_norm_k, residual_handle, ls_options);  
        catch ME_ls; warning('Iter %d: Error executing line search function: %s', iter, ME_ls.message); success = false; break; end  
        fprintf('     Line search completed (%.2f seconds).\n', toc(tic_ls));  

        % 9. Check line search results and record  
        if ~success_ls  
            warning('Iter %d: Line search failed!', iter);   
            success = false;   
            res_norm = res_norm_k;   
            break;  
        end  
        fprintf('     Line search successful: alpha = %.2e, new residual norm [old] = %.4e\n', alpha, res_norm);  
        residual_history = [residual_history; res_norm];  

        % 10. Check for stagnation  
        if alpha < ls_options.min_alpha * 1.1; fprintf('     Warning: Step size alpha small (%.2e).\n', alpha); end  

    end % --- End Newton while loop ---  
    total_iteration_time = toc(tic_iteration);  
    fprintf('--- Newton (approx) iteration loop ended (total time %.2f seconds) ---\n', total_iteration_time);  

    if iter > 0; iterations_taken = iter; final_residual = res_norm; end % Update final status  
end % --- End of "if res_norm > tol" block ---  

% --- 12. Output final state and results ---  
fprintf('\n--- Newton (approx) iteration summary ---\n');  
if final_residual <= tol && success  
    fprintf('[Success] Newton(approx)+LS+DirectBC solver converged after %d iterations!\n', iterations_taken);  
    fprintf('  Viscosity coefficient mu = %.4e\n', mu);  
    fprintf('  Final residual norm [old] = %.4e (tolerance tol = %.4e)\n', final_residual, tol);  
else  
    fprintf('[Failure] Newton(approx)+LS+DirectBC solver did not converge within %d iterations or failed in between.\n', iterations_taken);  
    fprintf('  Viscosity coefficient mu = %.4e\n', mu);  
    fprintf('  Final residual norm [old] = %.4e (tolerance tol = %.4e)\n', final_residual, tol);  
    if ~success; fprintf('  Reason for failure: Error during iteration or line search failure.\n');  
    elseif iter == iter_max && final_residual > tol; fprintf('  Reason for failure: Maximum iteration count reached (%d).\n', iter_max); end  
    warning('Solver did not converge or failed! Results may be unreliable.');  
end  

% --- 13. Post-processing and visualization (consistent with Oseen version) ---  
fprintf('Extracting and plotting final results for mu=%.2f (Delaunay)...\n', mu);  
u_computed = x(1:Npb);   
v_computed = x(Npb+1:2*Npb);   
if issparse(u_computed); u_computed = full(u_computed); end;   
if issparse(v_computed); v_computed = full(v_computed); end  
nodes_inside_holes_p2 = false(Npb, 1);   
for i = 1:num_holes;   
    xc = holes(i,1);   
    yc = holes(i,2);   
    r = holes(i,3);   
    nodes_inside_holes_p2 = nodes_inside_holes_p2 | (dcircle(Pb, xc, yc, r) < -1e-2);   
end;   
u_computed(nodes_inside_holes_p2) = NaN;   
v_computed(nodes_inside_holes_p2) = NaN;  

fprintf('Generating Delaunay triangulation for velocity plotting...\n');  
try;   
    tri_v = delaunay(Pb(:,1), Pb(:,2));   
    delaunay_ok = true;   
catch;   
    warning('Delaunay failed for mu=%.2f', mu);   
    delaunay_ok = false;   
end  
if delaunay_ok  
    figure('Name', sprintf('U Vel (Stokes Lin. mu=%.2f - Delaunay)', mu));   
    try;   
        trisurf(tri_v, Pb(:,1), Pb(:,2), u_computed);   
        title(sprintf('U (mu=%.2f)', mu));   
        xlabel('x'); ylabel('y');   
        colorbar;   
        shading interp;   
        view(2);   
        axis equal;   
        axis([0 L 0 D]);   
    catch ME;   
        warning('Plot U failed: %s', ME.message);   
    end  
    figure('Name', sprintf('V Vel (Stokes Lin. mu=%.2f - Delaunay)', mu));   
    try;   
        trisurf(tri_v, Pb(:,1), Pb(:,2), v_computed);   
        title(sprintf('V (mu=%.2f)', mu));   
        xlabel('x'); ylabel('y');   
        colorbar;   
        shading interp;   
        view(2);   
        axis equal;   
        axis([0 L 0 D]);   
    catch ME;   
        warning('Plot V failed: %s', ME.message);   
    end  
    figure('Name', sprintf('Vel Mag (Stokes Lin. mu=%.2f - Delaunay)', mu));   
    try;   
        velocity_magnitude = sqrt(u_computed.^2 + v_computed.^2);   
        trisurf(tri_v, Pb(:,1), Pb(:,2), velocity_magnitude);   
        title(sprintf('|u| (mu=%.2f)', mu));   
        xlabel('x'); ylabel('y');   
        colorbar;   
        shading interp;   
        view(2);   
        axis equal;   
        axis([0 L 0 D]);   
    catch ME;   
        warning('Plot Mag failed: %s', ME.message);   
    end  
else;   
    fprintf('Skipping Delaunay plots for mu=%.2f\n', mu);   
end  
figure('Name', sprintf('Streamlines (Stokes Lin. mu=%.2f)', mu));   
try;   
    [X_reg, Y_reg] = meshgrid(linspace(0, L, 60), linspace(0, D, 40));   
    interp_u = scatteredInterpolant(Pb(:,1), Pb(:,2), u_computed, 'linear', 'none');   
    interp_v = scatteredInterpolant(Pb(:,1), Pb(:,2), v_computed, 'linear', 'none');   
    U_reg = interp_u(X_reg, Y_reg);   
    V_reg = interp_v(X_reg, Y_reg);   
    inside_holes_reg = false(size(X_reg));   
    for i = 1:num_holes;   
        points_reg = [X_reg(:), Y_reg(:)];   
        dist_vals_reg = dcircle(points_reg, holes(i,1), holes(i,2), holes(i,3));   
        inside_hole_i_reg = reshape(dist_vals_reg < -1e-8, size(X_reg));   
        inside_holes_reg = inside_holes_reg | inside_hole_i_reg;   
    end;   
    U_reg(inside_holes_reg) = NaN;   
    V_reg(inside_holes_reg) = NaN;   
    hold on;   
    streamslice(X_reg, Y_reg, U_reg, V_reg, 1.5);   
    plot([0 L L 0 0], [0 0 D D 0], 'k-', 'LineWidth', 1);   
    for i = 1:num_holes;   
        viscircles(holes(i,1:2), holes(i,3), 'Color', 'k', 'LineWidth', 1);   
    end;   
    hold off;   
    title(sprintf('Streamlines (mu=%.2f)', mu));   
    xlabel('x'); ylabel('y');   
    axis equal;   
    axis([0 L 0 D]);   
catch ME;   
    warning('Plot Streamlines failed: %s', ME.message);   
end  
fprintf('Identifying boundary nodes for plotting...\n');   
try;   
    Dbc1_plot = index_val_Dirichlet_BC_channel(P,T,Pb,Tb,0,D,holes,g1_inlet,g1_wall);  
    Dbc2_plot_nodes = index_val_Dirichlet_BC_channel(P,T,Pb,Tb,0,D,holes,g2_all_dirichlet,g2_all_dirichlet);   
    plot_nodes_u = [];   
    if ~isempty(Dbc1_plot); plot_nodes_u = Dbc1_plot(:,1); end;   
    plot_nodes_v = [];   
    if ~isempty(Dbc2_plot_nodes); plot_nodes_v = Dbc2_plot_nodes(:,1); end;   
    plot_nodes_all_vel = unique([plot_nodes_u; plot_nodes_v]);   
    plot_nodes_on_holes = [];   
    if ~isempty(plot_nodes_all_vel);   
        coords_boundary = Pb(plot_nodes_all_vel,:);   
        is_on_a_hole = false(length(plot_nodes_all_vel),1);   
        hole_plot_tol = 1e-2;   
        for i = 1:length(plot_nodes_all_vel);   
            for h = 1:num_holes;   
                if abs(dcircle(coords_boundary(i,:), holes(h,1), holes(h,2), holes(h,3))) < hole_plot_tol;   
                    is_on_a_hole(i) = true; break;   
                end;   
            end;   
        end;   
        plot_nodes_on_holes = plot_nodes_all_vel(is_on_a_hole);   
        plot_nodes_outer = setdiff(plot_nodes_all_vel, plot_nodes_on_holes);   
    else;   
        plot_nodes_on_holes = [];   
        plot_nodes_outer = [];   
    end;   
    fprintf('Distinguished %d outer & %d hole boundary nodes for plot.\n', length(plot_nodes_outer), length(plot_nodes_on_holes));   
    figure('Name', sprintf('Identified Boundary Nodes (Stokes Lin. mu=%.4f)', mu));   
    triplot(T, P(:,1), P(:,2), 'Color', [0.7 0.7 0.7]);   
    hold on;   
    if ~isempty(plot_nodes_outer); plot(Pb(plot_nodes_outer, 1), Pb(plot_nodes_outer, 2), 'b.', 'MarkerSize', 4, 'DisplayName', 'Outer'); end;   
    if ~isempty(plot_nodes_on_holes); plot(Pb(plot_nodes_on_holes, 1), Pb(plot_nodes_on_holes, 2), 'ro', 'MarkerSize', 5, 'MarkerFaceColor', 'r', 'DisplayName', 'Holes'); end;   
    title(sprintf('Identified P2 Boundary Nodes (Red=Holes), mu=%.4f', mu));   
    axis equal;   
    axis([0 L 0 D]);   
    xlabel('x'); ylabel('y');   
    legend show;   
    for j = 1:num_holes;   
        viscircles(holes(j,1:2), holes(j,3), 'Color', 'k', 'LineStyle', '--');   
    end;   
    hold off;   
catch ME_bcplot; warning('Could not plot boundary nodes: %s', ME_bcplot.message); end  

% --- 14. Plot residual history (consistent with Oseen version) ---  
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

end % --- End main function ---  