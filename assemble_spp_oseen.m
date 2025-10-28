function A_oseen = assemble_spp_oseen(A_vv, B_div, An1_contribution, type)


fprintf('Assembling Oseen saddle-point matrix...\n');

% --- Input Dimension Checks ---
if ~issparse(A_vv) || ~issparse(B_div) || ~issparse(An1_contribution)
     warning('One or more input matrices are not sparse. Converting.');
     A_vv = sparse(A_vv);
     B_div = sparse(B_div);
     An1_contribution = sparse(An1_contribution);
end

Npb = size(A_vv, 1) / 2; % Infer number of velocity nodes per component
Np = size(B_div, 1);     % Number of pressure nodes

if size(A_vv, 1) ~= 2*Npb || size(A_vv, 2) ~= 2*Npb
    error('Dimension mismatch: A_vv size is %dx%d, expected %dx%d.', size(A_vv,1), size(A_vv,2), 2*Npb, 2*Npb);
end
if size(B_div, 2) ~= 2*Npb
    error('Dimension mismatch: B_div size is %dx%d, expected %dx%d.', size(B_div,1), size(B_div,2), Np, 2*Npb);
end
if size(An1_contribution, 1) ~= 2*Npb || size(An1_contribution, 2) ~= 2*Npb
    error('Dimension mismatch: An1_contribution size is %dx%d, expected %dx%d.', size(An1_contribution,1), size(An1_contribution,2), 2*Npb, 2*Npb);
end
% --- End Checks ---


% --- Combine Diffusion and Convection Blocks ---
A_vv_oseen = A_vv + An1_contribution;
fprintf('Combined A_vv and An1_contribution.\n');

% --- Construct the Saddle-Point Matrix ---
% Assuming 'sym' type implies the standard [Sys, B'; B, 0] structure
if strcmpi(type, 'sym')
    A_oseen = [ A_vv_oseen   B_div'          ;
                B_div      sparse(Np, Np) ];
    fprintf('Assembled symmetric-style Oseen matrix.\n');
elseif strcmpi(type, 'asym')
     % Adjust if your 'asym' type uses different signs or non-transposed B
     warning('Asymmetric type handling might need specific adjustments for Oseen.');
     A_oseen = [ A_vv_oseen   B_div'          ; % Still assuming B' typically
                 -B_div     sparse(Np, Np) ]; % Example with negative B
else
    error('Unknown type: %s. Use ''sym'' or ''asym''.', type);
end

fprintf('Oseen saddle-point matrix assembly complete. Size: %d x %d\n', size(A_oseen,1), size(A_oseen,2));

end % End of function assemble_spp_oseen