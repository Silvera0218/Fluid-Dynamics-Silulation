function An2 = assemble_An2_v(P, T, Pb, Tb, gauss_bary, weight, p_fem, u_k_vec)
    Npb = size(Pb, 1);
    Ne = size(Tb, 1);
    Nlb = size(Tb, 2);
    Ng = size(gauss_bary, 1);

    % --- Input Checks and Preparation ---
    if isvector(u_k_vec) && length(u_k_vec) == 2 * Npb; u_k = [u_k_vec(1:Npb), u_k_vec(Npb+1:2*Npb)];
    elseif size(u_k_vec, 1) == Npb && size(u_k_vec, 2) == 2; u_k = u_k_vec;
    else; error('Input velocity field u_k_vec has incorrect format'); end
    
    gauss_xy = gauss_bary(:,);
    phi_ref = basis_function(p_fem, 0, 0, gauss_xy);
    dphix_ref = basis_function(p_fem, 1, 0, gauss_xy);
    dphiy_ref = basis_function(p_fem, 0, 1, gauss_xy);

    % --- Sparse Matrix Assembly Preparation ---
    max_entries_total = 4 * Ne * Nlb * Nlb;
    ii_An2 = zeros(max_entries_total, 1); jj_An2 = zeros(max_entries_total, 1); ss_An2 = zeros(max_entries_total, 1);
    entry_count_total = 0;

    % --- Loop over Elements ---
    for k = 1:Ne
        nodes_k = Tb(k, :);
        P_k_vertices = P(T(k,:), :);
        x1=P_k_vertices(1,1); y1=P_k_vertices(1,2); x2=P_k_vertices(2,1); y2=P_k_vertices(2,2); x3=P_k_vertices(3,1); y3=P_k_vertices(3,2);
        detJ = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1);
        if abs(detJ) < 1e-12; continue; end
        abs_detJ = abs(detJ);
        invJ11=(y3-y1)/detJ; invJ12=-(x3-x1)/detJ; invJ21=-(y2-y1)/detJ; invJ22=(x2-x1)/detJ;

        u_k_local_nodes = u_k(nodes_k, :);
        dphix_phys = invJ11 * dphix_ref + invJ21 * dphiy_ref;
        dphiy_phys = invJ12 * dphix_ref + invJ22 * dphiy_ref;

        duk1_dx_at_gauss = u_k_local_nodes(:, 1)' * dphix_phys;
        duk1_dy_at_gauss = u_k_local_nodes(:, 1)' * dphiy_phys;
        duk2_dx_at_gauss = u_k_local_nodes(:, 2)' * dphix_phys;
        duk2_dy_at_gauss = u_k_local_nodes(:, 2)' * dphiy_phys;

        % --- Vectorized computation of local matrix blocks ---
        phi_outer_phi_times_weight_detJ = zeros(Nlb, Nlb, Ng);
        for q=1:Ng
            phi_q = phi_ref(:, q);
            phi_outer_phi_times_weight_detJ(:,:,q) = (phi_q * phi_q') * weight(q) * abs_detJ;
        end

        Ck_11 = sum(phi_outer_phi_times_weight_detJ .* reshape(duk1_dx_at_gauss, 1, 1, Ng), 3);
        Ck_12 = sum(phi_outer_phi_times_weight_detJ .* reshape(duk1_dy_at_gauss, 1, 1, Ng), 3);
        Ck_21 = sum(phi_outer_phi_times_weight_detJ .* reshape(duk2_dx_at_gauss, 1, 1, Ng), 3);
        Ck_22 = sum(phi_outer_phi_times_weight_detJ .* reshape(duk2_dy_at_gauss, 1, 1, Ng), 3);

        % --- Store triplets for sparse assembly ---
        for i = 1:Nlb; for j = 1:Nlb
                global_row_i = nodes_k(i); global_col_j = nodes_k(j);
                if abs(Ck_11(i,j)) > 1e-14; entry_count_total = entry_count_total + 1; ii_An2(entry_count_total) = global_row_i;      jj_An2(entry_count_total) = global_col_j;      ss_An2(entry_count_total) = Ck_11(i,j); end
                if abs(Ck_12(i,j)) > 1e-14; entry_count_total = entry_count_total + 1; ii_An2(entry_count_total) = global_row_i;      jj_An2(entry_count_total) = global_col_j + Npb; ss_An2(entry_count_total) = Ck_12(i,j); end
                if abs(Ck_21(i,j)) > 1e-14; entry_count_total = entry_count_total + 1; ii_An2(entry_count_total) = global_row_i + Npb; jj_An2(entry_count_total) = global_col_j;      ss_An2(entry_count_total) = Ck_21(i,j); end
                if abs(Ck_22(i,j)) > 1e-14; entry_count_total = entry_count_total + 1; ii_An2(entry_count_total) = global_row_i + Npb; jj_An2(entry_count_total) = global_col_j + Npb; ss_An2(entry_count_total) = Ck_22(i,j); end
        end; end
    end

    % --- Assemble Global Sparse Matrix An2 ---
    if entry_count_total > 0
        ii_An2 = ii_An2(1:entry_count_total); jj_An2 = jj_An2(1:entry_count_total); ss_An2 = ss_An2(1:entry_count_total);
        An2 = sparse(ii_An2, jj_An2, ss_An2, 2*Npb, 2*Npb);
    else
        An2 = sparse(2*Npb, 2*Npb);
    end
end
