function main_unsteady_newton_approx_fixediter_directbc()
% Solves Unsteady NS using Approximate Newton (Image Formula)
% Uses FIXED number of inner iterations per time step.
% NO Line Search. NO Residual Calculation/Check for convergence.
% Uses Direct BC Elimination.
% =========================================================================


clear all; close all; clc;

fprintf('Solving Unsteady Navier-Stokes (Channel Flow - Approx Newton + FixedIter + DirectBC [NO LS/RES CHECK])...\n');
fprintf('===================================================================================================\n');
fprintf('!!!!! WARNING: Running without line search and residual check !!!!!\n');

% --- 1. Parameter Settings ---
mu = 0.1; fprintf('[Parameter] Viscosity mu = %.4e\n', mu);
p_fem = 2; type = 'sym'; fprintf('[Parameter] Velocity element order p_fem = %d\n', p_fem);
T_final = 16.0; dt = 0.8; Nt = ceil(T_final / dt); inv_dt = 1.0 / dt;
fprintf('[Time] Final Time T = %.2f, Time Step dt = %.4f, Num Steps Nt = %d\n', T_final, dt, Nt);
iter_max_inner = 4; % << FIXED number of inner iterations per time step >>
fprintf('[Solver] FIXED Inner Iterations per step = %d\n', iter_max_inner);
if iter_max_inner <= 0; error('iter_max_inner must be positive.'); end
f1 = @(x,y) zeros(size(x)); f2 = @(x,y) zeros(size(x));

% --- 2. Load Mesh & Generate P2 Info ---
fprintf('Loading mesh and parameters...\n');
mesh_file = 'domain_mesh.mat'; if ~exist(mesh_file, 'file'); error('Mesh file %s not found!', mesh_file); end
try; loaded_data = load(mesh_file, 'p', 't', 'L', 'D', 'holes'); P = loaded_data.p; T = loaded_data.t; L = loaded_data.L; D = loaded_data.D; holes = loaded_data.holes;
catch ME; error('Cannot load mesh file %s: %s', mesh_file, ME.message); end
Np = size(P, 1); try; [Pb, Tb] = FEmesh(P, T, p_fem); Npb = size(Pb, 1); catch ME_femesh; error('FEmesh function call failed: %s.', ME_femesh.message); end; num_holes = size(holes, 1);
fprintf('Mesh loaded: P1=%d nodes, P2=%d nodes.\n', Np, Npb);

% --- 3. Define BC Functions & Generate Dbc List ---
fprintf('Generating constraint list Dbc...\n');
g1_inlet = @(x,y) atan(20*(D/2-abs(D/2-y))); g1_wall = @(x,y) zeros(size(x)); g2_all_dirichlet=@(x,y) zeros(size(x)); g3_pressure_pin=@(x,y) 0;
try
    Dbc1 = index_val_Dirichlet_BC_channel(P, T, Pb, Tb, 0, D, holes, g1_inlet, g1_wall);
    Dbc2_nodes = index_val_Dirichlet_BC_channel(P, T, Pb, Tb, 0, D, holes, g2_all_dirichlet, g2_all_dirichlet); Dbc2 = []; if ~isempty(Dbc2_nodes); Dbc2 = [Dbc2_nodes(:,1) + Npb, Dbc2_nodes(:,2)]; end
    [~, p_pin_idx] = min(sqrt(sum((P - [L/2,D/2]).^2, 2))); p_pin_dof = 2*Npb + p_pin_idx; Dbc3 = [p_pin_dof, g3_pressure_pin(P(p_pin_idx,1), P(p_pin_idx,2))];
    Dbc = unique([Dbc1; Dbc2; Dbc3], 'rows');
    fprintf('Identified %d unique constraints.\n', size(Dbc, 1));
catch ME_dbc; error('Error generating Dbc list: %s.', ME_dbc.message); end

% --- 4. Assemble Time-Independent Matrices & Get Quadrature ---
fprintf('Assembling time-independent matrices...\n');
quad_order = 2*p_fem; [gauss_bary, weight] = gauss_integration(quad_order);
fprintf('Obtained %d Gauss points (barycentric coordinates).\n', size(gauss_bary,1));
tic_const = tic;
As_block = assemble_A_v(Pb, Tb, gauss_bary, weight, p_fem, mu);      % Stiffness
B_div = assemble_Bp_v(P, T, Pb, Tb, gauss_bary, weight, p_fem, 1); % Divergence
M_scalar = assemble_M_v(Pb, Tb, gauss_bary, weight, p_fem, 1.0); % Mass matrix
M_block = blkdiag(M_scalar, M_scalar);                               % Block mass matrix
fprintf('Constant matrix assembly complete (elapsed time %.2f s).\n', toc(tic_const));
b0 = sparse(2*Npb+Np, 1); % Zero external source term

% --- 5. Define Unsteady Nonlinear Residual Function Handle (Needed internally by some checks potentially, define anyway) ---
% Although not used for convergence check, keep definition in case needed elsewhere or for future debugging
residual_handle_unsteady = @(xt, x_old_step) residual_navier_stokes_unsteady_nonlinear(Pb, Tb, gauss_bary, weight, p_fem, As_block, B_div, M_block, inv_dt, xt, x_old_step, b0, Dbc);

% --- 6. Time Stepping Loop ---
fprintf('--- Starting Time Stepping Loop (Backward Euler) ---\n');
x = zeros(2*Npb+Np, 1); time = 0;

% --- Storage for results & Plotting Prep ---
results.time = zeros(Nt+1, 1); results.solution = zeros(length(x), Nt+1); results.iterations = zeros(Nt, 1); % Store inner iterations count
results.time(1) = time; results.solution(:,1) = x;
output_times = [10.5, 11.4]; output_indices = [];
nodes_inside_holes_p2 = false(Npb, 1); for i=1:num_holes; try; nodes_inside_holes_p2=nodes_inside_holes_p2 | (dcircle(Pb, holes(i,1), holes(i,2), holes(i,3)) < -1e-8); catch; break; end; end
delaunay_ok_plot = false; tri_v_plot = []; try; tri_v_plot = delaunay(Pb(:,1), Pb(:,2)); delaunay_ok_plot = true; catch; end
[X_reg, Y_reg] = meshgrid(linspace(0, L, 60), linspace(0, D, 40));
live_plot_fig = figure('Name', 'Unsteady Flow Simulation (Approx Newton - Fixed Iter)'); set(live_plot_fig, 'DoubleBuffer', 'on');

for n = 1:Nt % Time loop
    time = n * dt; x_old = x;
    fprintf('\n--- Time Step %d/%d (t = %.4f) ---\n', n, Nt, time);

    % --- Inner Newton (Approx) Iteration Loop (Fixed Number of Iterations) ---
    x_k = x_old; % Use x_old as the starting point for the inner loop
    success_inner = true; % Assume success unless an error occurs

    for iter_inner = 1:iter_max_inner % Run exactly iter_max_inner times
        x_prev_inner = x_k; % Store current inner solution before update
        v0 = [x_prev_inner(1:Npb), x_prev_inner(Npb+1:2*Npb)]; % v0 = u^{k-1}

        fprintf('  --- Inner Iter %d/%d ---\n', iter_inner, iter_max_inner);

        % 1. Assemble An1(v0) & An2(v0)
        try; An1 = assemble_An1_v(P, T, Pb, Tb, gauss_bary, weight, p_fem, v0); catch ME; warning('InnerIter %d: Error An1: %s',iter_inner,ME.message); success_inner=false; break; end
        try; An2 = assemble_An2_v(P, T, Pb, Tb, gauss_bary, weight, p_fem, v0); catch ME; warning('InnerIter %d: Error An2: %s',iter_inner,ME.message); success_inner=false; break; end

        % 2. Calculate Fn(v0) vector = N(v0)
        try; Fn1 = assemble_bc_v(Pb, Tb, gauss_bary, weight, p_fem, v0, v0(:,1)); Fn2 = assemble_bc_v(Pb, Tb, gauss_bary, weight, p_fem, v0, v0(:,2)); Fn_vec = [Fn1; Fn2; sparse(Np,1)]; catch ME; warning('InnerIter %d: Error Fn: %s',iter_inner,ME.message); success_inner=false; break; end

        % 3. Build Newton system matrix A_iter and RHS b_iter
        LHS_Matrix_VelVel = inv_dt * M_block + As_block + An1 + An2;
        A_iter = [LHS_Matrix_VelVel, -B_div'; B_div, sparse(Np,Np)];
        U_old_vec = x_old(1:2*Npb); rhs_time_term_vel = inv_dt * M_block * U_old_vec;
        b0_vel = b0(1:2*Npb); b0_p = b0(2*Npb+1:end); Fn_vel = Fn_vec(1:2*Npb);
        b_iter = sparse(2*Npb+Np, 1); b_iter(1:2*Npb) = b0_vel + rhs_time_term_vel + Fn_vel; b_iter(2*Npb+1:end) = b0_p;

        % 4. Apply direct elimination BCs
        try; [A_bc, b_bc] = add_Dirichlet_BC(A_iter, b_iter, Dbc); catch ME; warning('InnerIter %d: Error BCs: %s',iter_inner,ME.message); success_inner=false; break; end

        % 5. Solve linear system for x_k (the next iterate)
        try
            x_k = A_bc \ b_bc; % Directly update x_k
            if any(isnan(x_k)) || any(isinf(x_k)); warning('InnerIter %d: Solve NaN/Inf!',iter_inner); success_inner=false; break; end
        catch ME; warning('InnerIter %d: Error Solve: %s',iter_inner,ME.message); success_inner=false; break; end

        % --- NO RESIDUAL CHECK ---
        % --- NO LINE SEARCH ---

    end % --- End Inner Newton fixed iteration loop ---

    if ~success_inner % If any error occurred in the inner loop
         warning('Time Step %d (t=%.4f): Inner solver failed. Stopping simulation.', n, time);
         break; % Stop time stepping
    end

    % Update solution for the next time step
    x = x_k;

    % --- Store results ---
    results.time(n+1) = time; results.solution(:,n+1) = x; results.iterations(n) = iter_inner; % Store how many iters ran (iter_max_inner unless break)
    % results.final_residual(n) = NaN; % Residual not calculated for convergence

    fprintf('  Time step %d finished after %d fixed inner iterations.\n', n, iter_inner);

  % --- Live Plotting 修改 ---  
figure(live_plot_fig); clf;  

% 预处理数据，将圆圈内数据设为 NaN  
u_plot = x(1:Npb);   
v_plot = x(Npb+1:2*Npb);  

for i = 1:num_holes  
    inside_circle = (Pb(:,1) - holes(i,1)).^2 + (Pb(:,2) - holes(i,2)).^2 <= holes(i,3)^2;  
    u_plot(inside_circle) = NaN;  
    v_plot(inside_circle) = NaN;  
end  

% 增加插值网格分辨率  
[X_reg, Y_reg] = meshgrid(linspace(0, L, 200), linspace(0, D, 150));  

h_sp1 = subplot(1, 2, 1); % 流线  
set(h_sp1, 'Position', [0.05, 0.1, 0.4, 0.8]);  

try  
    % 使用 griddata 插值  
    U_reg = griddata(Pb(:,1), Pb(:,2), u_plot, X_reg, Y_reg, 'cubic');  
    V_reg = griddata(Pb(:,1), Pb(:,2), v_plot, X_reg, Y_reg, 'cubic');  
    
    % 处理圆孔区域  
    for i = 1:num_holes  
        inside_hole = (X_reg - holes(i,1)).^2 + (Y_reg - holes(i,2)).^2 <= holes(i,3)^2;  
        U_reg(inside_hole) = NaN;  
        V_reg(inside_hole) = NaN;  
    end  
    
    hold on;  
    streamslice(X_reg, Y_reg, U_reg, V_reg, 1.5);  
    plot([0 L L 0 0], [0 0 D D 0], 'k-', 'LineWidth', 1);  
    
    for i = 1:num_holes  
        viscircles(holes(i,1:2), holes(i,3), 'Color', 'k', 'LineWidth', 1);  
    end  
    
    hold off;  
    title(sprintf('Streamlines (t=%.2f)', time));  
    axis equal;  
    axis([0 L 0 D]);  
catch ME_stream  
    title(sprintf('Streamlines (t=%.2f) - Error', time));  
    axis equal;  
    axis([0 L 0 D]);  
    warning('Streamline plot failed at t=%.2f: %s', time, ME_stream.message);  
end  
xlabel('x');  
ylabel('y');  

h_sp2 = subplot(1, 2, 2); % 速度幅值  
set(h_sp2, 'Position', [0.55, 0.1, 0.4, 0.8]);  

try  
    % 计算速度幅值  
    velocity_magnitude = sqrt(u_plot.^2 + v_plot.^2);  
    
    % 使用 griddata 插值速度幅值  
    V_mag_reg = griddata(Pb(:,1), Pb(:,2), velocity_magnitude, X_reg, Y_reg, 'cubic');  
    
    % 将圆圈区域设为 NaN  
    for i = 1:num_holes  
        inside_hole = (X_reg - holes(i,1)).^2 + (Y_reg - holes(i,2)).^2 <= holes(i,3)^2;  
        V_mag_reg(inside_hole) = NaN;  
    end  
    
    % 使用 pcolor 绘制，空白区域透明  
    h = pcolor(X_reg, Y_reg, V_mag_reg);  
    set(h, 'AlphaData', ~isnan(V_mag_reg));  % 使 NaN 区域透明  
    shading interp;  
    colorbar;  
    
    title(sprintf('|u| (t=%.2f)', time));  
    axis equal;  
    axis([0 L 0 D]);  
    
    % 动态颜色范围  
    clim_max = max(velocity_magnitude(~isnan(velocity_magnitude)), [], 'all');  
    if isempty(clim_max) || ~isfinite(clim_max) || clim_max < 1e-6  
        clim_max = 1;  
    end  
    clim([0, min(clim_max*1.1, 5)]);  
    
catch ME_mag  
    title(sprintf('|u| (t=%.2f) - Error', time));  
    axis equal;  
    axis([0 L 0 D]);  
    warning('Vel Mag plot failed at t=%.2f: %s', time, ME_mag.message);  
end  
xlabel('x');  
ylabel('y');  

sgtitle(sprintf('Unsteady NS (mu=%.2f, dt=%.3f) - Time = %.2f / %.2f', mu, dt, time, T_final), 'FontSize', 10);  
drawnow;  

    % --- Save snapshot at specific times ---
    for ot = 1:length(output_times); if abs(time - output_times(ot)) < dt/2 && ~ismember(n, output_indices); fprintf('*** Saving snapshot at t=%.2f (Step %d) ***\n', time, n); snapshot_filename = sprintf('snapshot_newtonFI_t%.2f_mu%.2f.png', output_times(ot), mu); try; saveas(live_plot_fig, snapshot_filename); catch; warning('Could not save snapshot %s', snapshot_filename); end; output_indices = [output_indices, n]; break; end; end

    if ~success_inner; break; end % Stop time stepping if inner solver failed

end % --- End Time Stepping Loop ---

% --- Final Output ---
fprintf('\n--- Unsteady Simulation Summary (Approx Newton Fixed Iter) ---\n');
fprintf('Finished at t=%.4f after %d/%d time steps.\n', time, n, Nt);
if n == Nt && success_inner; fprintf('Simulation completed (ran %d fixed inner iterations per step).\n', iter_max_inner); else; fprintf('Simulation stopped early or inner solver failed.\n'); end
figure;   
plot(1:n, results.iterations(1:n), '.-');  
xlabel('Time Step Number');  
ylabel('Inner Iterations Done');  
title(sprintf('Inner Newton Iterations per Time Step (Fixed=%d)',iter_max_inner));  
grid on;  
ylim([0, max(results.iterations(1:n))+1]); % 正确的 ylim 使用方式  



    % Create animation of the results  
    figure('Name', 'Unsteady Flow Simulation - Full Animation', 'Position', [100, 100, 1200, 600]);  
    
    % Prepare for animation  
    num_steps = size(results.solution, 2);  
    Nodes_inside_holes_p2 = false(Npb, 1);  
    for i = 1:num_holes  
        try  
            Nodes_inside_holes_p2 = Nodes_inside_holes_p2 | (dcircle(Pb, holes(i,1), holes(i,2), holes(i,3)) < -1e-8);  
        catch  
            break;  
        end  
    end  
    
    % Create animation  
    animation_filename = 'unsteady_flow_simulation_101.gif';  
    for n = 1:num_steps  
        clf;  
        
        % Extract solution for this time step  
        x = results.solution(:,n);  
        u_plot = x(1:Npb);  
        v_plot = x(Npb+1:2*Npb);  
        u_plot(Nodes_inside_holes_p2) = NaN;  
        v_plot(Nodes_inside_holes_p2) = NaN;  
        
        % Subplot 1: Streamlines  
        h_sp1 = subplot(1, 2, 1);  
        try  
            % Interpolation for streamlines (similar to live plotting)  
            interp_u = scatteredInterpolant(Pb(:,1), Pb(:,2), u_plot, 'linear', 'none');  
            interp_v = scatteredInterpolant(Pb(:,1), Pb(:,2), v_plot, 'linear', 'none');  
            U_reg = interp_u(X_reg, Y_reg);  
            V_reg = interp_v(X_reg, Y_reg);  
            
            % Handle holes  
            inside_holes_reg = false(size(X_reg));  
            for i = 1:num_holes  
                points_reg = [X_reg(:), Y_reg(:)];  
                dist_vals_reg = dcircle(points_reg, holes(i,1), holes(i,2), holes(i,3));  
                inside_hole_i_reg = reshape(dist_vals_reg < -1e-8, size(X_reg));  
                inside_holes_reg = inside_holes_reg | inside_hole_i_reg;  
            end  
            U_reg(inside_holes_reg) = NaN;  
            V_reg(inside_holes_reg) = NaN;  
            
            hold on;  
            streamslice(X_reg, Y_reg, U_reg, V_reg, 1.5);  
            plot([0 L L 0 0], [0 0 D D 0], 'k-', 'LineWidth', 1);  
            for i = 1:num_holes  
                viscircles(holes(i,1:2), holes(i,3), 'Color', 'k', 'LineWidth', 1);  
            end  
            hold off;  
            title(sprintf('Streamlines (t=%.2f)', results.time(n)));  
            axis equal;  
            axis([0 L 0 D]);  
        catch ME  
            title(sprintf('Streamlines (t=%.2f) - Error', results.time(n)));  
            axis equal;  
            axis([0 L 0 D]);  
            warning('Streamline plot failed: %s', ME.message);  
        end  
        xlabel('x');  
        ylabel('y');  
        
        % Subplot 2: Velocity Magnitude  
        h_sp2 = subplot(1, 2, 2);  
        try  
            velocity_magnitude = sqrt(u_plot.^2 + v_plot.^2);  
            if delaunay_ok_plot  
                trisurf(tri_v_plot, Pb(:,1), Pb(:,2), velocity_magnitude);  
            else  
                scatter(Pb(:,1), Pb(:,2), 10, velocity_magnitude, 'filled');  
            end  
            title(sprintf('|u| (t=%.2f)', results.time(n)));  
            colorbar;  
            shading interp;  
            view(2);  
            axis equal;  
            axis([0 L 0 D]);  
            clim_max = max(velocity_magnitude(~isnan(velocity_magnitude)), [], 'all');  
            if isempty(clim_max) || ~isfinite(clim_max) || clim_max < 1e-6  
                clim_max = 1;  
            end  
            clim([0, min(clim_max*1.1, 5)]);  
        catch ME  
            title(sprintf('|u| (t=%.2f) - Error', results.time(n)));  
            axis equal;  
            axis([0 L 0 D]);  
            warning('Vel Mag plot failed: %s', ME.message);  
        end  
        xlabel('x');  
        ylabel('y');  
        
        % Make subplot sizes equal  
        set(h_sp1, 'Position', [0.05, 0.1, 0.4, 0.8]);  
        set(h_sp2, 'Position', [0.55, 0.1, 0.4, 0.8]);  
        
        sgtitle(sprintf('Unsteady NS (mu=%.2f, dt=%.3f) - Time = %.2f / %.2f', mu, dt, results.time(n), T_final), 'FontSize', 10);  
        
        % Create GIF  
        drawnow;  
        frame = getframe(gcf);  
        im = frame2im(frame);  
        [imind, cm] = rgb2ind(im, 256);  
        
        % Write to the GIF file  
        if n == 1  
            imwrite(imind, cm, animation_filename, 'gif', 'Loopcount', inf, 'DelayTime', 0.1);  
        else  
            imwrite(imind, cm, animation_filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);  
        end  
    end  
    
    fprintf('Animation saved as %s\n', animation_filename);  



end % --- End main function ---