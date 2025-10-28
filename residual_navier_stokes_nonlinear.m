function res = residual_navier_stokes_nonlinear(Pb, Tb, gauss_bary, weight, p_fem, As_block, B_div, xt, b0, Dbc)
% 计算稳态、非线性 Navier-Stokes 方程的残差向量 (标准版本)
% Residual = [ As_block*U + N(U) - B_div'*P ; B_div*U ] - b0
% 用于 Newton 法的 RHS 计算 (-Residual) 和线搜索/收敛判断。
% 在计算范数前或用于 RHS 前，将 Dirichlet 节点的残差置零。
%
% 输入:
%   Pb, Tb:   P2 节点坐标和单元拓扑
%   gauss_bary: 参考单元上的高斯积分点 (重心坐标, Ng x 3)
%   weight:   高斯积分权重 (1 x Ng 或 Ng x 1)
%   p_fem:    速度基函数次数 (应为 2)
%   As_block: 速度扩散矩阵块 blkdiag(A_laplace, A_laplace)
%   B_div:    散度矩阵 B
%   xt:       当前试探解向量 [U; V; P] (维度: 2*Npb + Np)
%   b0:       源项向量 (通常为零)
%   Dbc:      Dirichlet 边界条件列表 [DOF_index, value]
%
% 输出:
%   res:      残差向量 (维度: 2*Npb + Np)，Dirichlet 节点对应分量已置零

    Npb = size(As_block, 1) / 2; % 每个速度分量的 P2 节点数
    Np = size(B_div, 1);       % P1 压力节点数
    Ntotal = 2*Npb + Np;
    Ng = size(gauss_bary, 1);

    % --- 检查输入维度 ---
    if length(xt) ~= Ntotal; error('Residual Function: 输入向量 xt 维度不正确'); end
    if length(b0) ~= Ntotal; error('Residual Function: 输入向量 b0 维度不正确'); end
    if size(As_block, 1) ~= 2*Npb || size(As_block, 2) ~= 2*Npb; error('Residual Function: As_block 维度不正确'); end
    if size(B_div, 2) ~= 2*Npb; error('Residual Function: B_div 维度不正确'); end
    if ~isempty(Dbc) && size(Dbc, 2) ~= 2; error('Residual Function: Dbc 格式不正确'); end
    if size(gauss_bary, 2) ~= 3; error('Residual Function: 需要重心坐标输入 (Ng x 3)'); end
    if size(weight, 1) == 1 && size(weight, 2) == Ng; % Ok, row vector
    elseif size(weight, 2) == 1 && size(weight, 1) == Ng; weight = weight'; % Convert col to row
    else; error('Residual Function: 权重 weight 维度应为 1 x Ng 或 Ng x 1'); end
    % --- 维度检查结束 ---

    % --- 提取速度和压力分量 ---
    U_vec = xt(1:2*Npb);         % 速度向量 (包含 U 和 V)
    P_vec = xt(2*Npb+1:end);     % 压力向量
    U = xt(1:Npb);             % 速度 U 分量
    V = xt(Npb+1:2*Npb);         % 速度 V 分量
    vel_field = [U, V];          % Npb x 2 格式，用于 assemble_bc_v

    % --- 组装完整的非线性对流项 N(xt) = [Nu; Nv] ---
    % !!! 调用 assemble_bc_v 时，传入重心坐标 gauss_bary !!!
    % !!! assemble_bc_v 必须在内部处理坐标转换 !!!
    try
        Nu = assemble_bc_v(Pb, Tb, gauss_bary, weight, p_fem, vel_field, U);
        Nv = assemble_bc_v(Pb, Tb, gauss_bary, weight, p_fem, vel_field, V);
        N_xt = [Nu; Nv]; % 非线性对流项向量
    catch ME_bc
        error('Residual Function: 调用 assemble_bc_v 计算对流项时出错: %s', ME_bc.message);
    end

    % --- 提取源项 b0 的速度和压力部分 ---
    b0_vel = b0(1:2*Npb);      % 对应动量方程的源项
    b0_p   = b0(2*Npb+1:end);   % 对应连续性方程的源项 (通常为零)

    % --- 计算残差分量 (控制方程的不平衡量) ---
    % 动量方程残差: As_block*U + N(U) - B_div'*P - b0_vel
    res_mom = As_block * U_vec + N_xt - B_div' * P_vec - b0_vel;

    % 连续性方程残差: B_div*U - b0_p
    res_cont = B_div * U_vec - b0_p;

    % --- 组合残差向量 ---
    res = [res_mom; res_cont];

    % --- 将 Dirichlet 边界自由度对应的残差分量置零 ---
    % 这是标准做法，因为这些自由度的值是由边界条件强制设定的，
    % 我们关心的是内部节点是否满足控制方程。
    if ~isempty(Dbc)
        dirichlet_dofs = Dbc(:,1); % 获取 Dirichlet 自由度的索引

        % 确保索引有效
        valid_indices = dirichlet_dofs > 0 & dirichlet_dofs <= length(res);
        if ~all(valid_indices)
             num_invalid = sum(~valid_indices);
             warning('Residual Function: Dbc 包含 %d 个无效的 DOF 索引，已忽略。最大 DOF 索引为 %d。', num_invalid, length(res));
             dirichlet_dofs = dirichlet_dofs(valid_indices); % 只保留有效索引
        end

        % 将对应位置的残差设置为 0
        res(dirichlet_dofs) = 0;
        % fprintf('DEBUG: Zeroed %d residual entries corresponding to Dirichlet DoFs.\n', length(dirichlet_dofs)); % Optional debug message
    end

end % 函数结束 residual_navier_stokes_nonlinear