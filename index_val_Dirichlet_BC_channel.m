function Dbc = index_val_Dirichlet_BC_channel(P, T, Pb, Tb, left, D_top, holes, g_inlet_func, g_wall_func)

fprintf('Identifying boundary nodes using Edge-Based method...\n');
tic;

% --- Input Validation ---
Ne = size(T, 1);
Np = size(P, 1);
Npb = size(Pb, 1);
if size(T,2) ~= 3; error('T must be Ne x 3 for P1 triangles.'); end
if size(Tb,2) ~= 6; error('Tb must be Ne x 6 for P2 triangles.'); end
if size(Pb,2) ~= 2; error('Pb must be Npb x 2.'); end
if size(P,2) ~= 2; error('P must be Np x 2.'); end
% --- End Validation ---

% --- Step 1: Find all unique edges and count occurrences ---
fprintf('Step 1: Finding unique edges from P1 mesh T...\n');
edges = [T(:,[1,2]); T(:,[2,3]); T(:,[3,1])]; % All edges (vertices)
% Sort node indices within each edge to handle directionality
edges = sort(edges, 2);
% Find unique edges and their counts
[unique_edges, ~, edge_map] = unique(edges, 'rows');
edge_counts = histcounts(edge_map, 1:(size(unique_edges, 1) + 1));

% --- Step 2: Identify boundary edges (count = 1) ---
boundary_edge_indices = find(edge_counts == 1);
boundary_edges_p1 = unique_edges(boundary_edge_indices, :); % Edges defined by P1 node indices
num_boundary_edges = size(boundary_edges_p1, 1);
fprintf('Found %d unique P1 boundary edges.\n', num_boundary_edges);

if num_boundary_edges == 0
    warning('No boundary edges found! Check mesh connectivity T.');
    Dbc = []; return;
end

% --- Step 3: Identify all P2 nodes on these boundary edges ---
% Start with P1 nodes on boundary edges
p1_boundary_nodes = unique(boundary_edges_p1(:));

% Find P2 midpoint nodes associated with these boundary edges
p2_midpoint_boundary_nodes = [];
fprintf('Finding P2 midpoint nodes on boundary edges...\n');
map_edge_to_midpoint = containers.Map('KeyType','char','ValueType','double');
% Build map from edge (v1,v2) -> midpoint index m12 from Tb
for k = 1:Ne
    nodes = Tb(k,:); % [v1 v2 v3 m12 m23 m31]
    map_edge_to_midpoint(sprintf('%d_%d', min(nodes(1),nodes(2)), max(nodes(1),nodes(2)))) = nodes(4);
    map_edge_to_midpoint(sprintf('%d_%d', min(nodes(2),nodes(3)), max(nodes(2),nodes(3)))) = nodes(5);
    map_edge_to_midpoint(sprintf('%d_%d', min(nodes(3),nodes(1)), max(nodes(3),nodes(1)))) = nodes(6);
end

% Iterate through P1 boundary edges and find corresponding midpoint
for i = 1:num_boundary_edges
    v1 = boundary_edges_p1(i, 1);
    v2 = boundary_edges_p1(i, 2);
    edge_key = sprintf('%d_%d', min(v1,v2), max(v1,v2));
    if map_edge_to_midpoint.isKey(edge_key)
        p2_midpoint_boundary_nodes = [p2_midpoint_boundary_nodes; map_edge_to_midpoint(edge_key)];
    else
        warning('Could not find midpoint for P1 boundary edge %d-%d in Tb mapping.', v1, v2);
    end
end
p2_midpoint_boundary_nodes = unique(p2_midpoint_boundary_nodes);

% Combine P1 vertex nodes and P2 midpoint nodes on the boundary
all_p2_boundary_nodes = unique([p1_boundary_nodes; p2_midpoint_boundary_nodes]);
fprintf('Found %d unique P1 boundary vertices and %d unique P2 boundary midpoints.\n', length(p1_boundary_nodes), length(p2_midpoint_boundary_nodes));
fprintf('Total %d unique P2 nodes identified on boundaries.\n', length(all_p2_boundary_nodes));


% --- Step 4: Classify the identified P2 boundary nodes ---
fprintf('Step 4: Classifying P2 boundary nodes...\n');
coord_tol = 1e-7;        % Tolerance for straight boundaries
dcircle_tol_classify = 1e-4; % Tolerance for classifying hole nodes (adjust if needed)
bottom = 0;
num_holes = size(holes, 1);

nodes_inlet_final = [];
nodes_wall_final = []; % Includes top, bottom, holes

boundary_nodes_coords = Pb(all_p2_boundary_nodes, :); % Get coordinates only for boundary nodes

for i = 1:length(all_p2_boundary_nodes)
    idx = all_p2_boundary_nodes(i); % Global P2 node index
    coord = boundary_nodes_coords(i,:);
    is_classified_as_wall = false;

    % Check if on Inlet (Highest priority)
    if abs(coord(1) - left) < coord_tol
        nodes_inlet_final = [nodes_inlet_final; idx];
        continue; % Skip other checks if identified as inlet
    end

    % Check if on Top Wall
    if abs(coord(2) - D_top) < coord_tol
        is_classified_as_wall = true;
    end

    % Check if on Bottom Wall
    if abs(coord(2) - bottom) < coord_tol
         is_classified_as_wall = true;
    end

    % Check if on any Hole boundary using dcircle
    if ~is_classified_as_wall % Check holes only if not top/bottom
        for h = 1:num_holes
             if abs(dcircle(coord, holes(h,1), holes(h,2), holes(h,3))) < dcircle_tol_classify
                 is_classified_as_wall = true;
                 break; % Stop checking holes
             end
        end
    end

    % Assign to wall list if classified
    if is_classified_as_wall
         nodes_wall_final = [nodes_wall_final; idx];
    % else: Node is likely on outlet, ignore for Dirichlet BCs
    end
end

% Ensure uniqueness and remove potential overlaps
nodes_inlet_final = unique(nodes_inlet_final);
nodes_wall_final = unique(nodes_wall_final);
nodes_wall_final = setdiff(nodes_wall_final, nodes_inlet_final); % Remove corners if needed
fprintf('Classified %d inlet nodes and %d wall nodes.\n', length(nodes_inlet_final), length(nodes_wall_final));


% --- Step 5: Apply Boundary Conditions based on classification ---
fprintf('Step 5: Applying boundary conditions...\n');
Dbc_inlet = [];
if ~isempty(nodes_inlet_final)
    coords_inlet = Pb(nodes_inlet_final, :);
    try; vals_inlet = g_inlet_func(coords_inlet(:,1), coords_inlet(:,2)); Dbc_inlet = [nodes_inlet_final, vals_inlet(:)]; catch ME; error('Error evaluating g_inlet_func: %s', ME.message); end
end
Dbc_wall = [];
if ~isempty(nodes_wall_final)
    coords_wall = Pb(nodes_wall_final, :);
     try; vals_wall = g_wall_func(coords_wall(:,1), coords_wall(:,2)); Dbc_wall = [nodes_wall_final, vals_wall(:)]; catch ME; error('Error evaluating g_wall_func: %s', ME.message); end
end
Dbc_combined = [Dbc_inlet; Dbc_wall];
if isempty(Dbc_combined); Dbc = []; warning('No Dirichlet constraints generated.'); else; Dbc = unique(Dbc_combined, 'rows'); fprintf('Generated Dbc matrix with %d unique constraints.\n', size(Dbc, 1)); end

figure('Name', '边界节点识别 (Edge-Based)');
plot(Pb(:,1), Pb(:,2), '.', 'MarkerSize', 1, 'Color', [0.8 0.8 0.8]); hold on;
if ~isempty(nodes_inlet_final); plot(Pb(nodes_inlet_final,1), Pb(nodes_inlet_final,2), 'bo', 'MarkerSize', 3, 'DisplayName', 'Inlet'); end
if ~isempty(nodes_wall_final); plot(Pb(nodes_wall_final,1), Pb(nodes_wall_final,2), 'rs', 'MarkerSize', 3, 'DisplayName', 'Walls'); end
L_plot = 17; % Assume L if not passed
plot([left left L_plot L_plot left], [bottom D_top D_top bottom bottom], 'k-', 'LineWidth', 1);
for j = 1:size(holes, 1); viscircles(holes(j,1:2), holes(j,3), 'Color', 'k', 'LineWidth', 0.5, 'LineStyle', '--'); end
axis equal; title('Classified Nodes (Edge-Based)'); legend show; hold off;


elapsed_time = toc;
fprintf('Edge-Based boundary identification finished in %.2f seconds.\n', elapsed_time);

end