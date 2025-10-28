function An1 = assemble_An1_v(P,T,Pb, Tb, gauss_bary, weight, p_fem, u_k_vec)
    Npb = size(Pb, 1);
    Ne = size(Tb, 1);
    Nlb = size(Tb, 2);
    Ng = size(gauss_bary, 1);

    % --- Input Checks and Preparation ---
    if size(gauss_bary, 2) ~= 3
        error('assemble_An1_v requires barycentric coordinates (Ng x 3)');
    end
    if size(weight, 1) ~= 1 || size(weight, 2) ~= Ng
         if size(weight, 2) == 1 && size(weight, 1) == Ng
             weight = weight'; % Ensure it's a row vector
         else
            error('Weight vector dimension should be 1 x Ng');
         end
    end
    if isvector(u_k_vec) && length(u_k_vec) == 2 * Npb
        u_k = [u_k_vec(1:Npb), u_k_vec(Npb+1:2*Npb)];
    elseif size(u_k_vec, 1) == Npb && size(u_k_vec, 2) == 2
        u_k = u_k_vec;
    else
        error('Input velocity field u_k_vec has incorrect format');
    end

    % --- Coordinate Transformation: Barycentric -> Reference Cartesian ---
    gauss_xy = gauss_bary(:,); % x = lambda2, y = lambda3

    % --- Get Basis Function Values and Derivatives ---
    phi_ref = basis_function(p_fem, 0, 0, gauss_xy); % Values @ Ng points, Nlb x Ng
    dphix_ref = basis_function(p_fem, 1, 0, gauss_xy); % Ref x-derivatives, Nlb x Ng
    dphiy_ref = basis_function(p_fem, 0, 1, gauss_xy); % Ref y-derivatives, Nlb x Ng

    % --- Sparse Matrix Assembly Preparation (Pre-allocation) ---
    max_entries = Ne * Nlb * Nlb;
    ii = zeros(max_entries, 1);
    jj = zeros(max_entries, 1);
    ss = zeros(max_entries, 1);
    entry_count = 0;

    % --- Loop over Elements for Assembly ---
    for k = 1:Ne
        nodes_k = Tb(k, :);
        P_k_vertices = P(T(k,:), :);

        x1=P_k_vertices(1,1); y1=P_k_vertices(1,2);
        x2=P_k_vertices(2,1); y2=P_k_vertices(2,2);
        x3=P_k_vertices(3,1); y3=P_k_vertices(3,2);
        detJ = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1);

        if abs(detJ) < 1e-12; continue; end
        abs_detJ = abs(detJ);

        invJ11 =  (y3-y1)/detJ; invJ12 = -(x3-x1)/detJ;
        invJ21 = -(y2-y1)/detJ; invJ22 =  (x2-x1)/detJ;

        u_k_local_nodes = u_k(nodes_k, :);
        u_k_at_gauss = u_k_local_nodes' * phi_ref;

        dphix_phys = invJ11 * dphix_ref + invJ21 * dphiy_ref;
        dphiy_phys = invJ12 * dphix_ref + invJ22 * dphiy_ref;

        Ck_local = zeros(Nlb, Nlb);
        for i = 1:Nlb
            for j = 1:Nlb
                uk_dot_grad_phij = u_k_at_gauss(1,:) .* dphix_phys(j,:) ...
                                 + u_k_at_gauss(2,:) .* dphiy_phys(j,:);
                integrand = uk_dot_grad_phij .* phi_ref(i,:);
                Ck_local(i,j) = (integrand .* weight) * ones(Ng, 1) * abs_detJ;
            end
        end

        row_indices_global = repmat(nodes_k', Nlb, 1);
        col_indices_global = repelem(nodes_k', Nlb, 1);
        num_local_entries = Nlb * Nlb;
        current_range = entry_count + (1:num_local_entries);

        ii(current_range) = row_indices_global;
        jj(current_range) = col_indices_global;
        ss(current_range) = Ck_local(:);
        entry_count = entry_count + num_local_entries;
    end

    % --- Assemble Global Sparse Matrix C ---
    ii = ii(1:entry_count);
    jj = jj(1:entry_count);
    ss = ss(1:entry_count);
    C = sparse(ii, jj, ss, Npb, Npb);

    % --- Construct Final Block Diagonal Matrix An1 ---
    An1 = blkdiag(C, C);
end
