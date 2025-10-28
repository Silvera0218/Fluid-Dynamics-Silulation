function main_oseen_ls_dbc_optimized()
    clear all; close all;
    
    fprintf('Solving Navier-Steady Stokes (Oseen + LS + Corrected Direct BC)...\n');

    params = setup_parameters();
    fprintf('*** Running simulation for mu = %.4f ***\n', params.mu);

    mesh = load_and_prepare_mesh('domain_mesh.mat', params.p_fem);

    Dbc = define_boundary_conditions(mesh, params);

    [A0_saddle, b0, gauss, weight] = assemble_stokes_operators(mesh, params);

    [x, residual_history, tol, status] = solve_oseen_linesearch(mesh, params, Dbc, A0_saddle, b0, gauss, weight);

    post_process_and_visualize(x, mesh, params, residual_history, tol, status);
    
    fprintf('\n--- Main function execution completed ---\n');
    fprintf('=============================================================================================\n');
end

function params = setup_parameters()
    params.mu = 1;
    params.p_fem = 2;
    params.type = 'sym';
    params.iter_max = 20;
    params.rtol = 1e-6;
    params.atol = 1e-8;
    params.f1 = @(x,y) zeros(size(x));
    params.f2 = @(x,y) zeros(size(x));
end

function mesh = load_and_prepare_mesh(filename, p_fem)
    fprintf('Loading mesh and parameters...\n');
    loaded_data = load(filename, 'p', 't', 'L', 'D', 'holes');
    mesh.P = loaded_data.p;
    mesh.T = loaded_data.t;
    mesh.L = loaded_data.L;
    mesh.D = loaded_data.D;
    mesh.holes = loaded_data.holes;
    mesh.Np = size(mesh.P, 1);
    mesh.num_holes = size(mesh.holes, 1);
    [mesh.Pb, mesh.Tb] = FEmesh(mesh.P, mesh.T, p_fem);
    mesh.Npb = size(mesh.Pb, 1);
    fprintf('Mesh Loaded: P1=%d, P2=%d\n', mesh.Np, mesh.Npb);
end

function Dbc = define_boundary_conditions(mesh, ~)
    fprintf('Generating constraints...\n');
    g1_inlet = @(x,y) atan(20*(mesh.D/2 - abs(mesh.D/2 - y)));
    g1_wall = @(x,y) zeros(size(x));
    g2_all_dirichlet = @(x,y) zeros(size(x));
    g3_pressure_pin = @(x,y) 0;

    Dbc1 = index_val_Dirichlet_BC_channel(mesh.P, mesh.T, mesh.Pb, mesh.Tb, 0, mesh.D, mesh.holes, g1_inlet, g1_wall);
    Dbc2_nodes = index_val_Dirichlet_BC_channel(mesh.P, mesh.T, mesh.Pb, mesh.Tb, 0, mesh.D, mesh.holes, g2_all_dirichlet, g2_all_dirichlet);
    
    Dbc2 = [];
    if ~isempty(Dbc2_nodes)
        Dbc2 = [Dbc2_nodes(:,1) + mesh.Npb, Dbc2_nodes(:,2)];
    end
    
    [~, p_pin_idx] = min(sqrt(sum((mesh.P - [mesh.L/2, mesh.D/2]).^2, 2)));
    p_pin_dof = 2*mesh.Npb + p_pin_idx;
    Dbc3 = [p_pin_dof, g3_pressure_pin(mesh.P(p_pin_idx,1), mesh.P(p_pin_idx,2))];
    
    Dbc = unique([Dbc1; Dbc2; Dbc3], 'rows');
    fprintf('Identified %d unique constraints.\n', size(Dbc, 1));
end

function [A0_saddle, b0, gauss, weight] = assemble_stokes_operators(mesh, params)
    fprintf('Assembling Stokes operator A0 & getting Gauss points...\n');
    quad_order = 2 * params.p_fem;
    [gauss, weight] = gauss_integration(quad_order);

    As_block = assemble_A_v(mesh.Pb, mesh.Tb, gauss, weight, params.p_fem, params.mu);
    B_div = assemble_Bp_v(mesh.P, mesh.T, mesh.Pb, mesh.Tb, gauss, weight, params.p_fem, 1);
    A0_saddle = assemble_spp(As_block, B_div, params.type);
    b0 = sparse(2*mesh.Npb + mesh.Np, 1);
end

function [x, residual_history, tol, status] = solve_oseen_linesearch(mesh, params, Dbc, A0_saddle, b0, gauss, weight)
    fprintf('--- Starting Oseen Iteration with Line Search ---\n');
    
    residual_handle = @(xt) residual_navier_stokes(mesh.Pb, mesh.Tb, gauss, weight, params.p_fem, A0_saddle, xt, b0, Dbc);
    
    x = zeros(2*mesh.Npb + mesh.Np, 1);
    res_iter = residual_handle(x);
    res_norm = norm(res_iter);
    res_0_norm = res_norm;
    residual_history = res_norm;
    tol = max(params.atol, res_0_norm * params.rtol);
    
    fprintf('Iter 0: Residual = %.4e\n', res_norm);
    
    ls_options.alpha_init = 1.0;
    ls_options.beta = 0.5;
    ls_options.c = 1e-8;
    ls_options.max_iter = 30;
    ls_options.min_alpha = 1e-8;
    
    iter = 0;
    while iter < params.iter_max && res_norm > tol
        iter = iter + 1;
        x_k = x;
        v0 = [x_k(1:mesh.Npb), x_k(mesh.Npb+1:2*mesh.Npb)];

        An1_contribution = assemble_An1_v(mesh.P, mesh.T, mesh.Pb, mesh.Tb, gauss, weight, params.p_fem, v0);
        A_iter = A0_saddle;
        A_iter(1:(2*mesh.Npb), 1:(2*mesh.Npb)) = A_iter(1:(2*mesh.Npb), 1:(2*mesh.Npb)) + An1_contribution;
        
        [A_bc, b_bc] = add_Dirichlet_BC(A_iter, b0, Dbc);
        
        x_solved = A_bc \ b_bc;
        delta_x = x_solved - x_k;
        
        res_norm_k = res_norm;
        [x, alpha, ~, res_norm] = perform_line_search(x_k, delta_x, res_norm_k, residual_handle, ls_options);
        
        fprintf('Iter %d: Residual = %.4e (Accepted alpha=%.2e)\n', iter, res_norm, alpha);
        residual_history = [residual_history; res_norm];
    end
    
    status.iterations = iter;
    status.final_residual = res_norm;
    status.converged = (res_norm <= tol);
end

function post_process_and_visualize(x, mesh, params, residual_history, tol, status)
    if status.converged
        fprintf('Oseen+LS+DirectBC solver converged in %d iterations for mu=%.4f.\n', status.iterations, params.mu);
    else
        warning('Oseen+LS+DirectBC solver DID NOT converge for mu=%.4f.', params.mu);
    end

    plot_delaunay_velocity(x, mesh, params);
    plot_streamlines(x, mesh, params);
    plot_boundary_nodes(mesh, params);
    plot_residual_history(residual_history, tol, params);
end

function plot_delaunay_velocity(x, mesh, params)
    fprintf('Plotting Delaunay velocity field...\n');
    u = full(x(1:mesh.Npb));
    v = full(x(mesh.Npb+1:2*mesh.Npb));
    
    nodes_inside = false(mesh.Npb, 1);
    for i = 1:mesh.num_holes
        nodes_inside = nodes_inside | (dcircle(mesh.Pb, mesh.holes(i,1), mesh.holes(i,2), mesh.holes(i,3)) < -1e-8);
    end
    u(nodes_inside) = NaN;
    v(nodes_inside) = NaN;
    
    try
        tri_v = delaunay(mesh.Pb(:,1), mesh.Pb(:,2));
    catch
        warning('Delaunay triangulation failed for mu=%.4f', params.mu);
        return;
    end
    
    figure('Name', sprintf('U Velocity (mu=%.4f)', params.mu));
    trisurf(tri_v, mesh.Pb(:,1), mesh.Pb(:,2), u);
    title(sprintf('U (mu=%.4f)', params.mu));
    xlabel('x'); ylabel('y'); colorbar; shading interp; view(2);
    axis equal; axis([0 mesh.L 0 mesh.D]);
    
    figure('Name', sprintf('V Velocity (mu=%.4f)', params.mu));
    trisurf(tri_v, mesh.Pb(:,1), mesh.Pb(:,2), v);
    title(sprintf('V (mu=%.4f)', params.mu));
    xlabel('x'); ylabel('y'); colorbar; shading interp; view(2);
    axis equal; axis([0 mesh.L 0 mesh.D]);
    
    figure('Name', sprintf('Velocity Magnitude (mu=%.4f)', params.mu));
    velocity_magnitude = sqrt(u.^2 + v.^2);
    trisurf(tri_v, mesh.Pb(:,1), mesh.Pb(:,2), velocity_magnitude);
    title(sprintf('|u| (mu=%.4f)', params.mu));
    xlabel('x'); ylabel('y'); colorbar; shading interp; view(2);
    axis equal; axis([0 mesh.L 0 mesh.D]);
end

function plot_streamlines(x, mesh, params)
    fprintf('Plotting streamlines...\n');
    u = full(x(1:mesh.Npb));
    v = full(x(mesh.Npb+1:2*mesh.Npb));
    
    try
        [X_reg, Y_reg] = meshgrid(linspace(0, mesh.L, 60), linspace(0, mesh.D, 40));
        interp_u = scatteredInterpolant(mesh.Pb(:,1), mesh.Pb(:,2), u, 'linear', 'none');
        interp_v = scatteredInterpolant(mesh.Pb(:,1), mesh.Pb(:,2), v, 'linear', 'none');
        U_reg = interp_u(X_reg, Y_reg);
        V_reg = interp_v(X_reg, Y_reg);
        
        inside_holes_reg = false(size(X_reg));
        for i = 1:mesh.num_holes
            points_reg = [X_reg(:), Y_reg(:)];
            dist_vals_reg = dcircle(points_reg, mesh.holes(i,1), mesh.holes(i,2), mesh.holes(i,3));
            inside_hole_i_reg = reshape(dist_vals_reg < -1e-8, size(X_reg));
            inside_holes_reg = inside_holes_reg | inside_hole_i_reg;
        end
        U_reg(inside_holes_reg) = NaN;
        V_reg(inside_holes_reg) = NaN;
        
        figure('Name', sprintf('Streamlines (mu=%.4f)', params.mu));
        hold on;
        streamslice(X_reg, Y_reg, U_reg, V_reg, 1.5);
        plot([0 mesh.L mesh.L 0 0], [0 0 mesh.D mesh.D 0], 'k-', 'LineWidth', 1);
        for i = 1:mesh.num_holes
            viscircles(mesh.holes(i,1:2), mesh.holes(i,3), 'Color', 'k', 'LineWidth', 1);
        end
        hold off;
        title(sprintf('Streamlines (mu=%.4f)', params.mu));
        xlabel('x'); ylabel('y'); axis equal; axis([0 mesh.L 0 mesh.D]);
    catch ME
        warning('Plotting streamlines failed: %s', ME.message);
    end
end

function plot_boundary_nodes(mesh, params)
    fprintf('Plotting identified boundary nodes...\n');
    try
        g1_inlet = @(x,y) atan(20*(mesh.D/2 - abs(mesh.D/2 - y)));
        g1_wall = @(x,y) zeros(size(x));
        g2_all_dirichlet = @(x,y) zeros(size(x));
        
        Dbc1_plot = index_val_Dirichlet_BC_channel(mesh.P, mesh.T, mesh.Pb, mesh.Tb, 0, mesh.D, mesh.holes, g1_inlet, g1_wall);
        Dbc2_plot_nodes = index_val_Dirichlet_BC_channel(mesh.P, mesh.T, mesh.Pb, mesh.Tb, 0, mesh.D, mesh.holes, g2_all_dirichlet, g2_all_dirichlet);
        
        plot_nodes_u = Dbc1_plot(:,1);
        plot_nodes_v = Dbc2_plot_nodes(:,1);
        plot_nodes_all_vel = unique([plot_nodes_u; plot_nodes_v]);
        
        coords_boundary = mesh.Pb(plot_nodes_all_vel,:);
        is_on_a_hole = false(length(plot_nodes_all_vel), 1);
        hole_plot_tol = 1e-2;
        for i = 1:length(plot_nodes_all_vel)
            for h = 1:mesh.num_holes
                if abs(dcircle(coords_boundary(i,:), mesh.holes(h,1), mesh.holes(h,2), mesh.holes(h,3))) < hole_plot_tol
                    is_on_a_hole(i) = true;
                    break;
                end
            end
        end
        plot_nodes_on_holes = plot_nodes_all_vel(is_on_a_hole);
        plot_nodes_outer = setdiff(plot_nodes_all_vel, plot_nodes_on_holes);
        
        fprintf('Distinguished %d outer & %d hole nodes for plot.\n', length(plot_nodes_outer), length(plot_nodes_on_holes));
        
        figure('Name', sprintf('Identified Boundary Nodes (mu=%.4f)', params.mu));
        triplot(mesh.T, mesh.P(:,1), mesh.P(:,2), 'Color', [0.7 0.7 0.7]);
        hold on;
        plot(mesh.Pb(plot_nodes_outer, 1), mesh.Pb(plot_nodes_outer, 2), 'b.', 'MarkerSize', 4, 'DisplayName', 'Outer');
        plot(mesh.Pb(plot_nodes_on_holes, 1), mesh.Pb(plot_nodes_on_holes, 2), 'ro', 'MarkerSize', 5, 'MarkerFaceColor', 'r', 'DisplayName', 'Holes');
        for j = 1:mesh.num_holes
            viscircles(mesh.holes(j,1:2), mesh.holes(j,3), 'Color', 'k', 'LineStyle', '--');
        end
        hold off;
        title(sprintf('Identified P2 Boundary Nodes (Red=Holes), mu=%.4f', params.mu));
        axis equal; axis([0 mesh.L 0 mesh.D]); xlabel('x'); ylabel('y'); legend show;
    catch ME
        warning('Could not plot boundary nodes: %s', ME.message);
    end
end

function plot_residual_history(residual_history, tol, params)
    if length(residual_history) <= 1
        fprintf('Insufficient residual history data, skipping plot.\n');
        return;
    end
    
    fprintf('Plotting residual history...\n');
    try
        figure('Name', sprintf('Residual History (mu=%.4e)', params.mu));
        semilogy(0:length(residual_history)-1, residual_history, '-o', 'LineWidth', 1.5, 'MarkerSize', 6);
        hold on;
        plot(get(gca, 'XLim'), [tol tol], 'r--', 'LineWidth', 1, 'DisplayName', sprintf('Tol=%.1e', tol));
        hold off;
        xlabel('Iteration count');
        ylabel('Residual norm (log scale)');
        title(sprintf('Convergence History (mu=%.4e)', params.mu));
        grid on;
        legend('Residual', 'Tolerance', 'Location', 'southwest');
    catch ME
        warning('Error plotting residual history: %s', ME.message);
    end
end
