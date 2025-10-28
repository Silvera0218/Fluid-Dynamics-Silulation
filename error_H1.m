function error_seminorm = error_H1(Nodes, Elements, p, gauss_bary, weight, u_coeffs, uexact_deriv_funs)
% Computes the H1 seminorm error ||grad(u_exact - u_h)||_L2 for 2D scalar fields.

    [Ne, Nlp] = size(Elements);
    Ng = size(gauss_bary, 1);
    Nnodes = size(Nodes, 1);

    if length(u_coeffs) ~= Nnodes
       error('Length of u_coeffs (%d) does not match number of nodes (%d).', length(u_coeffs), Nnodes);
    end
    expected_nlp = (p + 1) * (p + 2) / 2;
    if Ne > 0 && Nlp ~= expected_nlp
        warning('MATLAB:errorH1seminorm:NodeMismatch', ...
                'Number of nodes/element in Elements (%d) does not match expected for p=%d (%d).', ...
                Nlp, p, expected_nlp);
        Nlp = expected_nlp;
    end
    if ~iscell(uexact_deriv_funs) || length(uexact_deriv_funs) ~= 2
        error('uexact_deriv_funs must be a cell array {du/dx, du/dy}.');
    end
    uexact_dx_fun = uexact_deriv_funs{1};
    uexact_dy_fun = uexact_deriv_funs{2};

    if size(gauss_bary, 2) == 3
        gauss_ref_cart = gauss_bary(:, [2, 3]);
    elseif size(gauss_bary, 2) == 2
        gauss_ref_cart = gauss_bary;
    else
        error('Unsupported Gauss point format.');
    end

    try
        dphidxi_ref = basis_function(p, 1, 0, gauss_ref_cart);
        dphideta_ref = basis_function(p, 0, 1, gauss_ref_cart);
        if size(dphidxi_ref,1)~=Nlp || size(dphideta_ref,1)~=Nlp
            error('basis_function returned derivative matrix with incorrect rows.');
        end
    catch ME
        error('Error calling basis_function for derivatives (p=%d): %s', p, ME.message);
    end

    error_H1_seminorm_sq = 0;

    for ne = 1:Ne
        node_indices = Elements(ne, 1:Nlp);
        vertex_indices = Elements(ne, 1:3);
        vertex_coords = Nodes(vertex_indices, :)';

        J = [vertex_coords(:, 2) - vertex_coords(:, 1), vertex_coords(:, 3) - vertex_coords(:, 1)];
        detJ = det(J);
        if abs(detJ) < 1e-12, continue; end
        invJ = inv(J);
        v1_phys = vertex_coords(:, 1);

        for k = 1:Ng
            ref_coord_k = gauss_ref_cart(k, :)';
            weight_k = weight(k);
            factor = abs(detJ) * weight_k;

            phys_coord = J * ref_coord_k + v1_phys;
            x_k = phys_coord(1);
            y_k = phys_coord(2);

            dphixi_k = dphidxi_ref(:, k);
            dphieta_k = dphideta_ref(:, k);

            dphi_phys = invJ' * [dphixi_k'; dphieta_k'];
            dphi_dx_k = dphi_phys(1, :)';
            dphi_dy_k = dphi_phys(2, :)';

            try
                u_ex_dx_k = uexact_dx_fun(x_k, y_k);
                u_ex_dy_k = uexact_dy_fun(x_k, y_k);
                 if ~isscalar(u_ex_dx_k), if isscalar(u_ex_dx_k), u_ex_dx_k=u_ex_dx_k(1); else u_ex_dx_k=NaN; end; end
                 if ~isscalar(u_ex_dy_k), if isscalar(u_ex_dy_k), u_ex_dy_k=u_ex_dy_k(1); else u_ex_dy_k=NaN; end; end
            catch ME
                warning('Error evaluating exact derivatives @ (%g,%g): %s. Skipping point.', x_k, y_k, ME.message);
                 u_ex_dx_k=NaN; u_ex_dy_k=NaN;
            end

            try
                u_h_dx_k = u_coeffs(node_indices)' * dphi_dx_k;
                u_h_dy_k = u_coeffs(node_indices)' * dphi_dy_k;
            catch ME
                warning('Error calculating numerical derivatives @ element %d, point %d: %s. Skipping point.', ne, k, ME.message);
                 u_h_dx_k=NaN; u_h_dy_k=NaN;
            end

            if ~isnan(u_ex_dx_k) && ~isnan(u_h_dx_k) && ~isnan(u_ex_dy_k) && ~isnan(u_h_dy_k)
                error_H1_seminorm_sq = error_H1_seminorm_sq + factor * ((u_ex_dx_k - u_h_dx_k)^2 + (u_ex_dy_k - u_h_dy_k)^2);
            end

        end
    end

    error_seminorm = sqrt(error_H1_seminorm_sq);

end