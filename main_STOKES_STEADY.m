function main_stokes_channel_holes_remesh() % Steady Stokes, Penalty BC, Distmesh Remesh, Enhanced BC Plot

clear all; close all;

fprintf('Solving Steady Stokes Problem (Question 2.1) with Distmesh Remeshing...\n');


L = 17; D = 10;
holes = [ 3, 3, 1; 3, 7, 1; 7, 5, 1; 11, 3, 1; 11, 7, 1 ];
num_holes = size(holes, 1);
mu = 1; % Set viscosity
p_fem = 2; type = 'sym';


%2. Mesh Generation using Distmesh
fprintf('Generating mesh using Distmesh...\n');
fd_rect = @(p) drectangle(p, 0, L, 0, D);
if num_holes > 0;
    fd_all_holes = @(p) dcircle(p, holes(1,1), holes(1,2), holes(1,3)); 
    for i = 2:num_holes; fd_all_holes = @(p) dunion(fd_all_holes(p), dcircle(p, holes(i,1), holes(i,2), holes(i,3))); 
    end; 
    fd = @(p) ddiff(fd_rect(p), fd_all_holes(p)); 
else; 
    fd = fd_rect; end
bbox = [0, 0; L, D]; fh = @huniform; h0 = 0.2; fprintf('Using target mesh size h0 = %.3f\n', h0);
pfix = [0,0; L,0; L,D; 0,D; holes(:,1:2)];
fprintf('Calling distmesh2d (this may take some time)...\n'); tic;
try; [P, T] = distmesh2d(fd, fh, h0, bbox, pfix); mesh_time = toc; fprintf('Distmesh finished in %.2f seconds.\n', mesh_time); catch ME; error('Distmesh failed: %s', ME.message); end
Np = size(P, 1); fprintf('Generated P1 Mesh - Nodes: %d, Elements: %d\n', Np, size(T, 1));

% --- Define Dirichlet BC functions ---
g1_inlet = @(x,y) atan(20 * (D/2 - abs(D/2 - y))); g1_wall = @(x,y) zeros(size(x)); g2_all_dirichlet = @(x,y) zeros(size(x)); g3_pressure_pin = @(x,y) 0;

% --- Visualize P1 Mesh ---
figure('Name', 'Generated P1 Mesh (Distmesh)'); triplot(T, P(:,1), P(:,2)); title(sprintf('Generated P1 Mesh (h0=%.3f)', h0)); axis equal; axis([0 L 0 D]); xlabel('x'); ylabel('y'); hold on; plot(pfix(:,1), pfix(:,2), 'ro','MarkerSize',4); hold off;

% --- Generate P2 mesh info ---
[Pb, Tb] = FEmesh(P, T, p_fem); Npb = size(Pb, 1); fprintf('P1 Nodes: %d, P2 Nodes: %d\n', Np, Npb);

% --- 3. Assembly [A, b] ---
fprintf('Assembling Stokes system...\n');
quad_order = 2 * p_fem; [gauss, weight] = gauss_integration(quad_order);
As_block = assemble_A_v(Pb, Tb, gauss, weight, p_fem, mu);
B_div = assemble_Bp_v(P, T, Pb, Tb, gauss, weight, p_fem, 1);
A0 = assemble_spp(As_block, B_div, type); b0 = sparse(2*Npb+Np, 1);

% --- 4. Impose Dirichlet BCs using add_Dirichlet_BC_p function ---
fprintf('Identifying Dirichlet BCs using Edge-Based function...\n');

% --- Generate Dbc Matrix ---
% Ensure the boundary condition function is available
if ~exist('index_val_Dirichlet_BC_channel', 'file')
    error('Boundary condition function index_val_Dirichlet_BC_channel.m not found.');
end
% Call the edge-based boundary identifier
Dbc1 = index_val_Dirichlet_BC_channel(P, T, Pb, Tb, 0, D, holes, g1_inlet, g1_wall);
fprintf('Found %d constraints for U-component.\n', size(Dbc1, 1));

Dbc2_nodes = index_val_Dirichlet_BC_channel(P, T, Pb, Tb, 0, D, holes, g2_all_dirichlet, g2_all_dirichlet);
fprintf('Found %d constraints for V-component.\n', size(Dbc2_nodes, 1));
if ~isempty(Dbc2_nodes); Dbc2 = [Dbc2_nodes(:,1) + Npb, Dbc2_nodes(:,2)]; else; Dbc2 = []; end
% Pressure Pinning (near center)
center_point = [L/2, D/2]; distances_to_center = sqrt(sum((P - center_point).^2, 2));
[~, pressure_pin_node_idx] = min(distances_to_center);
fprintf('Pinning pressure at P1 node %d near center (Coords: %.2f, %.2f).\n', pressure_pin_node_idx, P(pressure_pin_node_idx,1), P(pressure_pin_node_idx,2));
pressure_pin_dof = 2*Npb + pressure_pin_node_idx;

Dbc3 = [pressure_pin_dof, g3_pressure_pin(P(pressure_pin_node_idx,1), P(pressure_pin_node_idx,2))];
% Combine and unique Dbc
Dbc = [Dbc1; Dbc2; Dbc3];
Dbc = unique(Dbc, 'rows');
fprintf('Identified %d unique Dirichlet constraints total.\n', size(Dbc, 1));
% --- End of Dbc generation ---


% --- Calculate Penalty Parameter ---
fprintf('Calculating Penalty Parameter...\n');
diag_elements = abs(diag(As_block)); % Use As_block for scaling penalty
diag_mag_sparse = max(diag_elements);
if issparse(diag_mag_sparse); diag_mag = full(diag_mag_sparse); else; diag_mag = diag_mag_sparse; end
if ~isscalar(diag_mag); error('diag_mag is still not a scalar!'); end
if diag_mag < eps; diag_mag = 1; end
penalty_factor = 1e9; % *** ADJUST THIS FACTOR AS NEEDED (e.g., 1e8, 1e10) ***
P_penalty = penalty_factor * diag_mag;
fprintf('Using Penalty value: %.2e (Factor=%.0e)\n', P_penalty, penalty_factor);
% --- End Penalty Parameter Calculation ---


% --- Apply Penalty BCs using the external function ---
fprintf('Calling add_Dirichlet_BC_p to apply penalty...\n');
if ~exist('add_Dirichlet_BC_p', 'file')
    error('Function add_Dirichlet_BC_p.m not found.');
end
[A, b] = add_Dirichlet_BC_p(A0, b0, Dbc, P_penalty); % *** CALL NEW FUNCTION ***
% --- End Apply Penalty ---

% --- 5. Solve: Ax = b ---
fprintf('Solving the linear system...\n');
if size(A,1) ~= size(b,1) || size(A,1) ~= size(A,2); error('Matrix/vector dimensions mismatch or A not square.'); end
tic; x = A \ b; solve_time = toc; fprintf('System solved in %.2f seconds.\n', solve_time);

% --- 6. Post-Processing and Visualization (Delaunay Vel, Skip Pressure, Enhanced BC Plot) ---
fprintf('Extracting and plotting results (setting NaN inside holes, skipping pressure)...\n');
u_computed = x(1:Npb); v_computed = x(Npb+1:2*Npb); if issparse(u_computed); u_computed=full(u_computed); end; if issparse(v_computed); v_computed=full(v_computed); end % Force full
% --- NaN Masking ---
nodes_inside_holes_p2 = false(Npb, 1); fprintf('Identifying P2 nodes inside holes...\n');
for i = 1:num_holes; xc = holes(i,1); yc = holes(i,2); r = holes(i,3); nodes_inside_holes_p2 = nodes_inside_holes_p2 | (dcircle(Pb, xc, yc, r) < -1e-8); end
num_nan_p2 = sum(nodes_inside_holes_p2); fprintf('Setting %d P2 nodes inside holes to NaN.\n', num_nan_p2);
u_computed(nodes_inside_holes_p2) = NaN; v_computed(nodes_inside_holes_p2) = NaN;
% --- End NaN Masking ---

% --- Generate Delaunay for Velocity ---
fprintf('Generating Delaunay triangulation for velocity plotting...\n');
try; tri_v = delaunay(Pb(:,1), Pb(:,2)); delaunay_ok = true; catch ME_delaunay; warning('Delaunay failed: %s', ME_delaunay.message); delaunay_ok = false; end

% --- Plot Velocity Components and Magnitude using Delaunay (tri_v) ---
if delaunay_ok
    fprintf('Plotting velocity results using Delaunay triangulation (tri_v)...\n');
    figure('Name', sprintf('U Vel (Delaunay - mu=%.4f)', mu)); try; trisurf(tri_v, Pb(:,1), Pb(:,2), u_computed); title(sprintf('U (mu=%.4f, Delaunay)', mu)); xlabel('x'); ylabel('y'); colorbar; shading interp; view(2); axis equal; axis([0 L 0 D]); catch ME; warning('Plot U (Delaunay) failed: %s',ME.message); end
    figure('Name', sprintf('V Vel (Delaunay - mu=%.4f)', mu)); try; trisurf(tri_v, Pb(:,1), Pb(:,2), v_computed); title(sprintf('V (mu=%.4f, Delaunay)', mu)); xlabel('x'); ylabel('y'); colorbar; shading interp; view(2); axis equal; axis([0 L 0 D]); catch ME; warning('Plot V (Delaunay) failed: %s',ME.message); end
    figure('Name', sprintf('Vel Mag (Delaunay - mu=%.4f)', mu)); try; velocity_magnitude = sqrt(u_computed.^2 + v_computed.^2); trisurf(tri_v, Pb(:,1), Pb(:,2), velocity_magnitude); title(sprintf('|u| (mu=%.4f, Delaunay)', mu)); xlabel('x'); ylabel('y'); colorbar; shading interp; view(2); axis equal; axis([0 L 0 D]); catch ME; warning('Plot Mag (Delaunay) failed: %s',ME.message); end
else
    fprintf('Skipping Delaunay plots for velocity as triangulation failed.\n');
end

% --- Plot Streamlines ---
figure('Name', sprintf('Streamlines (mu=%.4f)', mu)); try; [X_reg, Y_reg] = meshgrid(linspace(0, L, 60), linspace(0, D, 40)); interp_u = scatteredInterpolant(Pb(:,1), Pb(:,2), u_computed, 'linear', 'none'); interp_v = scatteredInterpolant(Pb(:,1), Pb(:,2), v_computed, 'linear', 'none'); U_reg = interp_u(X_reg, Y_reg); V_reg = interp_v(X_reg, Y_reg); inside_holes_reg = false(size(X_reg)); for i = 1:num_holes; points_reg = [X_reg(:), Y_reg(:)]; dist_vals_reg = dcircle(points_reg, holes(i,1), holes(i,2), holes(i,3)); inside_hole_i_reg = reshape(dist_vals_reg < -1e-8, size(X_reg)); inside_holes_reg = inside_holes_reg | inside_hole_i_reg; end; U_reg(inside_holes_reg) = NaN; V_reg(inside_holes_reg) = NaN; hold on; streamslice(X_reg, Y_reg, U_reg, V_reg, 1.5); plot([0 L L 0 0], [0 0 D D 0], 'k-', 'LineWidth', 1); for i = 1:num_holes; viscircles(holes(i,1:2), holes(i,3), 'Color', 'k', 'LineWidth', 1); end; hold off; title(sprintf('Velocity Streamlines (mu=%.4f)', mu)); xlabel('x'); ylabel('y'); axis equal; axis([0 L 0 D]); catch ME; warning('Plot Streamlines failed: %s', ME.message); end

% --- Add Enhanced Boundary Node Plotting ---
fprintf('Identifying boundary nodes for plotting...\n');
try
    % Re-run identification to get node indices
    Dbc1_plot = index_val_Dirichlet_BC_channel(P, T, Pb, Tb, 0, D, holes, g1_inlet, g1_wall);
    Dbc2_plot_nodes = index_val_Dirichlet_BC_channel(P, T, Pb, Tb, 0, D, holes, g2_all_dirichlet, g2_all_dirichlet);
    plot_nodes_u = []; if ~isempty(Dbc1_plot); plot_nodes_u = Dbc1_plot(:,1); end
    plot_nodes_v = []; if ~isempty(Dbc2_plot_nodes); plot_nodes_v = Dbc2_plot_nodes(:,1); end
    plot_nodes_all_vel = unique([plot_nodes_u; plot_nodes_v]);

    % Re-identify which of these boundary nodes are on holes
    plot_nodes_on_holes = [];
    if ~isempty(plot_nodes_all_vel)
        coords_boundary = Pb(plot_nodes_all_vel, :);
        is_on_a_hole = false(length(plot_nodes_all_vel), 1);
        hole_plot_tol = 1e-3; % Tolerance for classifying hole nodes for plot
        for i = 1:length(plot_nodes_all_vel)
            for h = 1:num_holes
                if abs(dcircle(coords_boundary(i,:), holes(h,1), holes(h,2), holes(h,3))) < hole_plot_tol
                    is_on_a_hole(i) = true;
                    break;
                end
            end
        end
        plot_nodes_on_holes = plot_nodes_all_vel(is_on_a_hole);
        plot_nodes_outer = setdiff(plot_nodes_all_vel, plot_nodes_on_holes);
    else
        plot_nodes_on_holes = []; plot_nodes_outer = [];
    end
    fprintf('Distinguished %d outer boundary nodes and %d hole boundary nodes for plotting.\n', length(plot_nodes_outer), length(plot_nodes_on_holes));

    % Create the plot
    figure('Name', sprintf('Identified Boundary Nodes on Mesh (mu=%.4f)', mu));
    triplot(T, P(:,1), P(:,2), 'Color', [0.7 0.7 0.7]); % Plot P1 mesh faintly
    hold on;
    if ~isempty(plot_nodes_outer); plot(Pb(plot_nodes_outer, 1), Pb(plot_nodes_outer, 2), 'b.', 'MarkerSize', 4, 'DisplayName', 'Outer Boundary'); end
    if ~isempty(plot_nodes_on_holes); plot(Pb(plot_nodes_on_holes, 1), Pb(plot_nodes_on_holes, 2), 'ro', 'MarkerSize', 5, 'MarkerFaceColor','r', 'DisplayName', 'Hole Boundaries'); end % Larger red markers
    title(sprintf('Identified P2 Boundary Nodes (Red=Holes), mu=%.4f', mu));
    axis equal; axis([0 L 0 D]); xlabel('x'); ylabel('y');
    legend show;
    for j = 1:size(holes, 1); viscircles(holes(j,1:2), holes(j,3), 'Color', 'k', 'LineWidth', 0.5, 'LineStyle', '--'); end % Ideal boundaries
    hold off;
catch ME_bcplot
    warning('Could not plot boundary nodes: %s', ME_bcplot.message);
end
% --- End Boundary Node Plotting ---


fprintf('Simulation complete.\n');

end % End of function