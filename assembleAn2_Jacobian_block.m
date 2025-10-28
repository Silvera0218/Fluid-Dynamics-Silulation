function [Kxx, Kxy, Kyx, Kyy] = assembleAn2_Jacobian_block(Pb, Tb, gauss_points_ref_cart, gauss_weights, p_fem, v0)
% Assembles ONLY the blocks Kxx, Kxy, Kyx, Kyy for the An2 convection term.
% An2 corresponds ONLY to the term arising from differentiating N(u) with
% respect to u in the second position: ((u_h · ∇)v0, v_h), where v0 is fixed.
% Kxx(i,j) = integral{ (phi_j * dv0x/dx) * phi_i }
% Kxy(i,j) = integral{ (phi_j * dv0x/dy) * phi_i }
% Kyx(i,j) = integral{ (phi_j * dv0y/dx) * phi_i }
% Kyy(i,j) = integral{ (phi_j * dv0y/dy) * phi_i }
% Assumes gauss_points_ref_cart are Ng x 2 reference Cartesian coordinates.
%
% Inputs:
%   Pb: Node coordinates for P2 elements (Npb x 2).
%   Tb: Element connectivity for P2 elements (Ne x 6).
%   gauss_points_ref_cart: Gauss points in REFERENCE CARTESIAN coords [xi, eta] (Ng x 2).
%   gauss_weights: Gauss weights (Ng x 1 or 1 x Ng).
%   p_fem: Polynomial degree (should be 2).
%   v0: Velocity vector at which the gradient is computed (Npb x 2).
%
% Output:
%   Kxx, Kxy, Kyx, Kyy: Assembled blocks for An2 (sparse, Npb x Npb).

fprintf('Assembling An2 Jacobian blocks ((u·∇)v0, v)...\n'); % Clarified name
tic;

Npb = size(Pb, 1);
[Ne, Nlb] = size(Tb);
Ng = size(gauss_weights, 1);

% --- Input Validation ---
if size(gauss_points_ref_cart, 2) ~= 2; error('Expected Ngx2 ref Cartesian coords.'); end
if p_fem ~= 2; warning('Designed for p_fem=2.'); end
if size(v0,1)~=Npb || size(v0,2)~=2; error('v0 size mismatch.'); end
if ~exist('basis_function','file'); error('basis_function.m not found.'); end
% --- End Validation ---

% Pre-calculate basis functions and derivatives
phi_val = basis_function(p_fem, 0, 0, gauss_points_ref_cart);   % Nlb x Ng
dphix_ref = basis_function(p_fem, 1, 0, gauss_points_ref_cart); % Nlb x Ng
dphiy_ref = basis_function(p_fem, 0, 1, gauss_points_ref_cart); % Nlb x Ng

% Sparse matrix assembly initialization
max_entries = Ne * Nlb * Nlb;
II = zeros(max_entries, 1); JJ = zeros(max_entries, 1);
SSxx = zeros(max_entries, 1); SSxy = zeros(max_entries, 1);
SSyx = zeros(max_entries, 1); SSyy = zeros(max_entries, 1);
entry_count = 0; % Use a single counter now

% Loop over elements
for k = 1:Ne
    local_nodes_indices = Tb(k, :);
    local_nodes_coords = Pb(local_nodes_indices, :);
    v0_local = v0(local_nodes_indices, :);

    Kxx_elem = zeros(Nlb, Nlb); Kxy_elem = zeros(Nlb, Nlb);
    Kyx_elem = zeros(Nlb, Nlb); Kyy_elem = zeros(Nlb, Nlb);

    % Loop over Gauss points
    for q = 1:Ng
        grad_phi_ref_q = [dphix_ref(:,q)'; dphiy_ref(:,q)']; % 2 x Nlb
        J = grad_phi_ref_q * local_nodes_coords; detJ = det(J);
        if detJ <= 1e-12; continue; end
        invJ = inv(J); invJ_T = invJ';
        grad_phi_global_q = invJ_T * grad_phi_ref_q; % Global gradients (2 x Nlb)

        % Calculate GRADIENT of v0 at Gauss point q: grad(v0)
        grad_v0_q = zeros(2, 2);
        for l = 1:Nlb
             grad_v0_q = grad_v0_q + v0_local(l,:)' * grad_phi_global_q(:,l)';
        end
        dv0x_dx = grad_v0_q(1,1); dv0x_dy = grad_v0_q(1,2);
        dv0y_dx = grad_v0_q(2,1); dv0y_dy = grad_v0_q(2,2);

        phi_val_q = phi_val(:, q); % Nlb x 1
        phi_i_phi_j_outer = phi_val_q * phi_val_q'; % Nlb x Nlb

        % Accumulate contributions to An2 blocks ONLY
        common_factor = gauss_weights(q) * abs(detJ);
        Kxx_elem = Kxx_elem + phi_i_phi_j_outer * dv0x_dx * common_factor;
        Kxy_elem = Kxy_elem + phi_i_phi_j_outer * dv0x_dy * common_factor;
        Kyx_elem = Kyx_elem + phi_i_phi_j_outer * dv0y_dx * common_factor;
        Kyy_elem = Kyy_elem + phi_i_phi_j_outer * dv0y_dy * common_factor;

    end % End Gauss points loop

    % Assemble local matrices into global COO storage
    row_indices = local_nodes_indices'; col_indices = local_nodes_indices;
    II_block = repmat(row_indices, 1, Nlb); JJ_block = repmat(col_indices, Nlb, 1);
    num_elem_entries = Nlb*Nlb; start_idx = entry_count + 1; end_idx = entry_count + num_elem_entries;

    % Store indices once, store values for each block
    II(start_idx:end_idx) = II_block(:);
    JJ(start_idx:end_idx) = JJ_block(:);
    SSxx(start_idx:end_idx) = Kxx_elem(:);
    SSxy(start_idx:end_idx) = Kxy_elem(:);
    SSyx(start_idx:end_idx) = Kyx_elem(:);
    SSyy(start_idx:end_idx) = Kyy_elem(:);
    entry_count = end_idx;

end % End elements loop

% Trim unused storage
II = II(1:entry_count); JJ = JJ(1:entry_count);

% Create sparse matrices
fprintf('Creating sparse An2 block matrices...\n');
Kxx = sparse(II, JJ, SSxx(1:entry_count), Npb, Npb);
Kxy = sparse(II, JJ, SSxy(1:entry_count), Npb, Npb);
Kyx = sparse(II, JJ, SSyx(1:entry_count), Npb, Npb);
Kyy = sparse(II, JJ, SSyy(1:entry_count), Npb, Npb);

elapsed_time = toc;
fprintf('Assembly of An2 blocks finished in %.2f seconds.\n', elapsed_time);

end % End of function