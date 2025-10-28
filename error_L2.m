function error = errorL2(Nodes, Elements, p, gauss_bary, weight, u_coeffs, uexact_fun)

    [Ne, Nlp] = size(Elements);
    Ng = size(gauss_bary, 1);
    error_sq = 0; 

  
    if size(gauss_bary, 2) == 3
        gauss_ref_cart = gauss_bary(:, [2, 3]); 
    elseif size(gauss_bary, 2) == 2
        gauss_ref_cart = gauss_bary; 
    else
        error('Cannot identify Gaussian point.');
    end

    phi_vals_ref = basis_function(p, 0, 0, gauss_ref_cart); 

    for ne = 1:Ne
        
        node_indices = Elements(ne, :); 
        vertex_indices = Elements(ne, 1:3); 

        vertex_coords = Nodes(vertex_indices, :)'; 

        J = [vertex_coords(:, 2) - vertex_coords(:, 1), vertex_coords(:, 3) - vertex_coords(:, 1)];
        detJ = det(J);
        if abs(detJ) < 1e-12, continue; end 
        v1_phys = vertex_coords(:, 1); 

        for k = 1:Ng

            ref_coord_k = gauss_ref_cart(k, :)'; 
            weight_k = weight(k);
            factor = abs(detJ) * weight_k;


            phys_coord = J * ref_coord_k + v1_phys; 
            x_k = phys_coord(1);
            y_k = phys_coord(2);

            phi_k = phi_vals_ref(:, k); 

            u_h_k = u_coeffs(node_indices)' * phi_k; 
            try
                u_ex_k = uexact_fun(x_k, y_k);
            catch ME
                error('Error on (x,y)=(%g,%g): %s', x_k, y_k, ME.message);
            end


            error_sq = error_sq + factor * (u_ex_k - u_h_k)^2;
        end 
    end 

    error = sqrt(error_sq);

end