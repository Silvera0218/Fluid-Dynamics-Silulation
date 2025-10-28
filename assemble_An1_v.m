
function An1 = assemble_An1_v(P,T,Pb, Tb, gauss_bary, weight, p_fem, u_k_vec)

    Npb = size(Pb, 1);
    Ne = size(Tb, 1);
    Nlb = size(Tb, 2);
    Ng = size(gauss_bary, 1);

    % --- 输入检查和准备 ---
    if size(gauss_bary, 2) ~= 3
        error('assemble_An1_v 需要重心坐标输入 (Ng x 3)');
    end
    if size(weight, 1) ~= 1 || size(weight, 2) ~= Ng
         if size(weight, 2) == 1 && size(weight, 1) == Ng
             weight = weight'; % 确保是行向量
         else
            error('权重 weight 维度应为 1 x Ng');
         end
    end
    if p_fem ~= round(p_fem) || p_fem <= 0
        error('p_fem 必须为正整数');
    end
    expected_nlb = (p_fem + 1) * (p_fem + 2) / 2;
    if Nlb ~= expected_nlb
        warning('Tb 的列数 (%d) 与 p_fem=%d (%d) 不匹配', Nlb, p_fem, expected_nlb);
        % 可能需要调整 Nlb 或检查 Tb/p_fem 是否正确
    end

    % 确保 u_k_vec 是 Npb x 2 矩阵格式
    if isvector(u_k_vec) && length(u_k_vec) == 2 * Npb
        u_k = [u_k_vec(1:Npb), u_k_vec(Npb+1:2*Npb)];
    elseif size(u_k_vec, 1) == Npb && size(u_k_vec, 2) == 2
        u_k = u_k_vec;
    else
        error('输入速度场 u_k_vec 格式不正确，应为 2*Npb x 1 或 Npb x 2');
    end
    % --- 输入检查结束 ---

    % --- 内部坐标转换: 重心坐标 -> 参考笛卡尔坐标 ---
    gauss_xy = gauss_bary(:, [2, 3]); % x = lambda2, y = lambda3
    % --- 转换结束 ---

    % --- 获取基函数值和导数 (使用转换后的 gauss_xy) ---
    phi_ref = basis_function(p_fem, 0, 0, gauss_xy); % 值 @ Ng points, Nlb x Ng
    dphix_ref = basis_function(p_fem, 1, 0, gauss_xy); % 参考 x 导数, Nlb x Ng
    dphiy_ref = basis_function(p_fem, 0, 1, gauss_xy); % 参考 y 导数, Nlb x Ng
    % --- 获取完毕 ---

    % 稀疏矩阵组装准备 (预分配内存)
    % 每个单元贡献 Nlb*Nlb 个非零元（近似）
    max_entries = Ne * Nlb * Nlb;
    ii = zeros(max_entries, 1); % 行索引
    jj = zeros(max_entries, 1); % 列索引
    ss = zeros(max_entries, 1); % 矩阵元素值
    entry_count = 0;

    fprintf('  内部组装 Oseen 矩阵 C (P%d)...\n', p_fem);
    tic_c = tic;

    % --- 遍历单元进行组装 ---
    for k = 1:Ne
        nodes_k = Tb(k, :);         % 当前单元的 P2 节点全局索引
        P_k_vertices = P(T(k,:), :); % 当前单元的 P1 顶点坐标 (用于计算 Jacobian)

        % 计算 Jacobian 和 逆 Jacobian (基于 P1 顶点线性映射)
        x1=P_k_vertices(1,1); y1=P_k_vertices(1,2);
        x2=P_k_vertices(2,1); y2=P_k_vertices(2,2);
        x3=P_k_vertices(3,1); y3=P_k_vertices(3,2);
        detJ = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1);

        if abs(detJ) < 1e-12 % 检查奇异或退化单元
            warning('单元 %d 的 Jacobian 行列式接近零 (%.2e)，跳过。', k, detJ);
            continue;
        end
        abs_detJ = abs(detJ);

        % 逆 Jacobian (用于变换导数)
        invJ11 =  (y3-y1)/detJ; invJ12 = -(x3-x1)/detJ;
        invJ21 = -(y2-y1)/detJ; invJ22 =  (x2-x1)/detJ;

        % --- 计算高斯点上的 u_k 值和物理导数 ---
        u_k_local_nodes = u_k(nodes_k, :);      % 获取当前单元节点上的 u_k 值 [Nlb x 2]
        u_k_at_gauss = u_k_local_nodes' * phi_ref; % 插值得到高斯点上的 u_k 值 [2 x Ng]

        % 计算物理导数 d(phi_j)/dx, d(phi_j)/dy @ Gauss points
        dphix_phys = invJ11 * dphix_ref + invJ21 * dphiy_ref; % [Nlb x Ng]
        dphiy_phys = invJ12 * dphix_ref + invJ22 * dphiy_ref; % [Nlb x Ng]
        % --- 计算完毕 ---

        % --- 计算单元矩阵 Ck ---
        % Ck_ij = sum_{g=1}^{Ng} [ weight(g) * ( (u_k(g) . grad(phi_j(g))) * phi_i(g) ) * abs_detJ ]
        Ck_local = zeros(Nlb, Nlb);
        for i = 1:Nlb % 遍历测试函数 phi_i
            for j = 1:Nlb % 遍历试探函数 phi_j
                % 计算点乘 (u_k . grad(phi_j)) 在所有高斯点的值
                uk_dot_grad_phij = u_k_at_gauss(1,:) .* dphix_phys(j,:) ...
                                 + u_k_at_gauss(2,:) .* dphiy_phys(j,:); % [1 x Ng]

                % 计算被积函数 integrand = (u_k . grad(phi_j)) * phi_i
                integrand = uk_dot_grad_phij .* phi_ref(i,:); % [1 x Ng]

                % 数值积分
                Ck_local(i,j) = (integrand .* weight) * ones(Ng, 1) * abs_detJ; % 等价于 sum(integrand .* weight) * abs_detJ
            end
        end
        % --- 单元矩阵计算完毕 ---

        % --- 存储用于稀疏组装的数据 ---
        row_indices_global = repmat(nodes_k', Nlb, 1); % 重复 Nlb 次列向量 nodes_k
        col_indices_global = repelem(nodes_k', Nlb, 1); % 每个元素重复 Nlb 次的列向量 nodes_k

        num_local_entries = Nlb * Nlb;
        current_range = entry_count + (1:num_local_entries); % 当前存储范围

        if current_range(end) > max_entries
             error('预分配内存不足，请检查 max_entries'); % 理论上不应发生
        end

        ii(current_range) = row_indices_global;
        jj(current_range) = col_indices_global;
        ss(current_range) = Ck_local(:); % 按列优先展平 Ck_local
        entry_count = entry_count + num_local_entries;
        % --- 存储完毕 ---
    end
    % --- 单元遍历结束 ---

    % --- 组装全局稀疏矩阵 C ---
    % 截断未使用的预分配空间
    ii = ii(1:entry_count);
    jj = jj(1:entry_count);
    ss = ss(1:entry_count);
    C = sparse(ii, jj, ss, Npb, Npb);
    assembly_time_c = toc(tic_c);
    fprintf('  内部 Oseen 矩阵 C 组装完成. Size: %d x %d, nnz: %d. Time: %.2f sec\n', Npb, Npb, nnz(C), assembly_time_c);

    % --- 构建最终的块对角矩阵 An1 ---
    An1 = blkdiag(C, C);
    fprintf('  最终 Oseen 块矩阵 An1 构建完成. Size: %d x %d.\n', size(An1,1), size(An1,2));

end % 函数结束 assemble_An1_v

