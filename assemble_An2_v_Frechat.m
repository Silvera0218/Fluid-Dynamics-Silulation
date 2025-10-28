function An2 = assemble_An2_v_Frechat(P, T, Pb, Tb, gauss_bary, weight, p_fem, u_k_vec)
% (标准 Newton 定义) 组装 An2 矩阵块，对应于 ((δu ⋅ ∇)u_k, v_h) 项。
% An2 = [C2_11, C2_12; C2_21, C2_22]
% C2_11(i,j) ≈ ∫ ( phi_j * ∂(uk1)/∂x ) * phi_i dΩ
% C2_12(i,j) ≈ ∫ ( phi_j * ∂(uk1)/∂y ) * phi_i dΩ
% C2_21(i,j) ≈ ∫ ( phi_j * ∂(uk2)/∂x ) * phi_i dΩ
% C2_22(i,j) ≈ ∫ ( phi_j * ∂(uk2)/∂y ) * phi_i dΩ
% 接收重心坐标 gauss_bary，内部转换为笛卡尔坐标 gauss_xy 后调用 basis_function。

    Npb = size(Pb, 1); Ne = size(Tb, 1); Nlb = size(Tb, 2); Ng = size(gauss_bary, 1);

    % --- 输入检查和准备 ---
    if size(gauss_bary, 2) ~= 3; error('需要重心坐标输入 (Ng x 3)'); end
    if size(weight, 1) ~= 1 || size(weight, 2) ~= Ng
         if size(weight, 2) == 1 && size(weight, 1) == Ng; weight = weight'; else; error('权重 weight 维度应为 1 x Ng'); end
    end
    if p_fem ~= round(p_fem) || p_fem <= 0; error('p_fem 必须为正整数'); end
    expected_nlb = (p_fem + 1) * (p_fem + 2) / 2;
    if Nlb ~= expected_nlb; warning('Tb 列数 (%d) 与 p_fem=%d (%d) 不匹配', Nlb, p_fem, expected_nlb); end
    if isvector(u_k_vec) && length(u_k_vec) == 2 * Npb; u_k = [u_k_vec(1:Npb), u_k_vec(Npb+1:2*Npb)];
    elseif size(u_k_vec, 1) == Npb && size(u_k_vec, 2) == 2; u_k = u_k_vec;
    else; error('输入速度场 u_k_vec 格式不正确'); end
    % --- 输入检查结束 ---

    % --- 内部坐标转换 ---
    gauss_xy = gauss_bary(:, [2, 3]);
    % --- 转换结束 ---

    % --- 获取基函数值和导数 ---
    phi_ref = basis_function(p_fem, 0, 0, gauss_xy);    % Nlb x Ng
    dphix_ref = basis_function(p_fem, 1, 0, gauss_xy);  % Nlb x Ng
    dphiy_ref = basis_function(p_fem, 0, 1, gauss_xy);  % Nlb x Ng
    % --- 获取完毕 ---

    % --- 稀疏矩阵组装准备 ---
    max_entries_total = 4 * Ne * Nlb * Nlb;
    ii_An2 = zeros(max_entries_total, 1); jj_An2 = zeros(max_entries_total, 1); ss_An2 = zeros(max_entries_total, 1);
    entry_count_total = 0;

    fprintf('  内部组装 Newton 矩阵 An2 (标准定义) (P%d)...\n', p_fem);
    tic_an2 = tic;

    % --- 遍历单元 ---
    for k = 1:Ne
        nodes_k = Tb(k, :);
        P_k_vertices = P(T(k,:), :);
        x1=P_k_vertices(1,1); y1=P_k_vertices(1,2); x2=P_k_vertices(2,1); y2=P_k_vertices(2,2); x3=P_k_vertices(3,1); y3=P_k_vertices(3,2);
        detJ = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1);
        if abs(detJ) < 1e-12; warning('单元 %d Jacobian 行列式接近零 (%.2e)，跳过。', k, detJ); continue; end
        abs_detJ = abs(detJ);
        invJ11=(y3-y1)/detJ; invJ12=-(x3-x1)/detJ; invJ21=-(y2-y1)/detJ; invJ22=(x2-x1)/detJ;

        u_k_local_nodes = u_k(nodes_k, :); % u_{k-1} 的节点值 [Nlb x 2]

        % 计算基函数的物理导数
        dphix_phys = invJ11 * dphix_ref + invJ21 * dphiy_ref; % [Nlb x Ng]
        dphiy_phys = invJ12 * dphix_ref + invJ22 * dphiy_ref; % [Nlb x Ng]

        % 计算上一步速度 u_k 在高斯点处的梯度
        duk1_dx_at_gauss = u_k_local_nodes(:, 1)' * dphix_phys; % [1 x Ng]
        duk1_dy_at_gauss = u_k_local_nodes(:, 1)' * dphiy_phys; % [1 x Ng]
        duk2_dx_at_gauss = u_k_local_nodes(:, 2)' * dphix_phys; % [1 x Ng]
        duk2_dy_at_gauss = u_k_local_nodes(:, 2)' * dphiy_phys; % [1 x Ng]

        % --- 计算单元矩阵块 (Standard Definition) ---
        Ck_11 = zeros(Nlb, Nlb); Ck_12 = zeros(Nlb, Nlb);
        Ck_21 = zeros(Nlb, Nlb); Ck_22 = zeros(Nlb, Nlb);

        # 预计算 phi_i * phi_j 在所有高斯点的值
        phi_outer_phi = zeros(Nlb, Nlb, Ng);
        for q = 1:Ng
            phi_q = phi_ref(:,q); % Nlb x 1
            phi_outer_phi(:,:,q) = phi_q * phi_q'; % Nlb x Nlb
        end

        # 向量化计算积分
        # Ck_ab(i,j) = sum_q [ weight(q) * (phi_i(q)*phi_j(q)) * duk_a/dx_b(q) * abs_detJ ]
        Ck_11 = sum(phi_outer_phi .* reshape(duk1_dx_at_gauss, 1, 1, Ng) .* reshape(weight, 1, 1, Ng), 3) * abs_detJ;
        Ck_12 = sum(phi_outer_phi .* reshape(duk1_dy_at_gauss, 1, 1, Ng) .* reshape(weight, 1, 1, Ng), 3) * abs_detJ;
        Ck_21 = sum(phi_outer_phi .* reshape(duk2_dx_at_gauss, 1, 1, Ng) .* reshape(weight, 1, 1, Ng), 3) * abs_detJ;
        Ck_22 = sum(phi_outer_phi .* reshape(duk2_dy_at_gauss, 1, 1, Ng) .* reshape(weight, 1, 1, Ng), 3) * abs_detJ;

        % --- 存储用于稀疏组装 An2 的数据 ---
        % (与上一版本相同，存储四个块的 triplets)
        for i = 1:Nlb; for j = 1:Nlb
                global_row_i = nodes_k(i); global_col_j = nodes_k(j);
                if abs(Ck_11(i,j)) > 1e-14; entry_count_total = entry_count_total + 1; ii_An2(entry_count_total) = global_row_i;      jj_An2(entry_count_total) = global_col_j;      ss_An2(entry_count_total) = Ck_11(i,j); end
                if abs(Ck_12(i,j)) > 1e-14; entry_count_total = entry_count_total + 1; ii_An2(entry_count_total) = global_row_i;      jj_An2(entry_count_total) = global_col_j + Npb; ss_An2(entry_count_total) = Ck_12(i,j); end
                if abs(Ck_21(i,j)) > 1e-14; entry_count_total = entry_count_total + 1; ii_An2(entry_count_total) = global_row_i + Npb; jj_An2(entry_count_total) = global_col_j;      ss_An2(entry_count_total) = Ck_21(i,j); end
                if abs(Ck_22(i,j)) > 1e-14; entry_count_total = entry_count_total + 1; ii_An2(entry_count_total) = global_row_i + Npb; jj_An2(entry_count_total) = global_col_j + Npb; ss_An2(entry_count_total) = Ck_22(i,j); end
        end; end
        % --- 存储完毕 ---

    end % --- 单元遍历结束 ---

    % --- 组装全局稀疏矩阵 An2 ---
    if entry_count_total == 0; warning('assemble_An2_v: 未找到非零元素'); An2 = sparse(2*Npb, 2*Npb);
    else
        ii_An2 = ii_An2(1:entry_count_total); jj_An2 = jj_An2(1:entry_count_total); ss_An2 = ss_An2(1:entry_count_total);
        An2 = sparse(ii_An2, jj_An2, ss_An2, 2*Npb, 2*Npb);
    end
    assembly_time_an2 = toc(tic_an2);
    fprintf('  内部 Newton 矩阵 An2 (标准定义) 组装完成. Size: %d x %d, nnz: %d. Time: %.2f sec\n', 2*Npb, 2*Npb, nnz(An2), assembly_time_an2);

end % 函数结束 assemble_An2_v