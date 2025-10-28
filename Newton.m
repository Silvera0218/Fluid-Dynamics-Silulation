function main_newton_frechet_ls_directbc() % Standard Newton (Frechet) + LS + Direct BC

clear all; close all; clc;

fprintf('求解稳态 Navier-Stokes (通道流 - 标准 Newton + LS + DirectBC [使用非线性残差])...\n');
fprintf('======================================================================================\n');

% --- 1. 参数设定 ---
mu = 0.01; % << 设定粘性系数 >>
fprintf('[参数] 粘性系数 mu = %.4e\n', mu);
p_fem = 2; type = 'sym'; fprintf('[参数] 速度单元次数 p_fem = %d\n', p_fem);
iter_max = 30; % Newton 收敛快时可能不需要很多次
rtol = 1e-7; atol = 1e-9; % 可以设置更严格的容忍度
fprintf('[参数] 最大迭代次数 iter_max = %d\n', iter_max);
fprintf('[参数] 相对容忍度 rtol = %.1e\n', rtol); fprintf('[参数] 绝对容忍度 atol = %.1e\n', atol);
f1 = @(x,y) zeros(size(x)); f2 = @(x,y) zeros(size(x));

% --- 2. 加载网格 & 生成 P2 信息 ---
% ... (省略详细代码, 与之前版本一致) ...
fprintf('加载网格和参数...\n'); mesh_file = 'domain_mesh.mat'; if ~exist(mesh_file, 'file'); error('网格文件 %s 未找到！', mesh_file); end; try; loaded_data = load(mesh_file, 'p', 't', 'L', 'D', 'holes'); P = loaded_data.p; T = loaded_data.t; L = loaded_data.L; D = loaded_data.D; holes = loaded_data.holes; catch ME; error('无法加载网格文件 %s: %s', mesh_file, ME.message); end; Np = size(P, 1); fprintf('生成 P%d 速度网格...\n', p_fem); try; [Pb, Tb] = FEmesh(P, T, p_fem); Npb = size(Pb, 1); catch ME_femesh; error('FEmesh 函数调用失败: %s.', ME_femesh.message); end; num_holes = size(holes, 1); fprintf('网格加载完成: P1=%d, P2=%d\n', Np, Npb);

% --- 3. 定义 BC 函数 & 生成 Dbc 列表 ---
% ... (省略详细代码, 与之前版本一致) ...
fprintf('生成约束列表 Dbc...\n'); g1_inlet = @(x,y) atan(20*(D/2-abs(D/2-y))); g1_wall = @(x,y) zeros(size(x)); g2_all_dirichlet=@(x,y) zeros(size(x)); g3_pressure_pin=@(x,y) 0; try; Dbc1 = index_val_Dirichlet_BC_channel(P, T, Pb, Tb, 0, D, holes, g1_inlet, g1_wall); Dbc2_nodes = index_val_Dirichlet_BC_channel(P, T, Pb, Tb, 0, D, holes, g2_all_dirichlet, g2_all_dirichlet); Dbc2 = []; if ~isempty(Dbc2_nodes); Dbc2 = [Dbc2_nodes(:,1) + Npb, Dbc2_nodes(:,2)]; end; [~, p_pin_idx] = min(sqrt(sum((P - [L/2,D/2]).^2, 2))); p_pin_dof = 2*Npb + p_pin_idx; Dbc3 = [p_pin_dof, g3_pressure_pin(P(p_pin_idx,1), P(p_pin_idx,2))]; Dbc = unique([Dbc1; Dbc2; Dbc3], 'rows'); fprintf('识别出 %d 个唯一约束。\n', size(Dbc, 1)); catch ME_dbc; error('生成 Dbc 列表时出错: %s.', ME_dbc.message); end

% --- 4. 组装常数矩阵 As_block, B_div & 获取高斯点 ---
fprintf('组装常数矩阵 As_block, B_div & 获取高斯点...\n');
quad_order = 2*p_fem; [gauss_bary, weight] = gauss_integration(quad_order);
% !!! 假设以下 assemble_* 函数处理内部坐标转换 !!!
As_block = assemble_A_v(Pb, Tb, gauss_bary, weight, p_fem, mu);      % 内部转换
B_div = assemble_Bp_v(P, T, Pb, Tb, gauss_bary, weight, p_fem, 1); % 内部转换
b0 = sparse(2*Npb+Np, 1); % 零源项
fprintf('常数矩阵和高斯点准备就绪。\n');

% --- 5. 定义非线性残差函数句柄 (使用标准版本) ---
% !!! residual_navier_stokes_nonlinear 需处理内部坐标转换 !!!
fprintf('定义非线性残差函数句柄 (使用 residual_navier_stokes_nonlinear)...\n');
residual_handle = @(xt) residual_navier_stokes_nonlinear(Pb, Tb, gauss_bary, weight, p_fem, As_block, B_div, xt, b0, Dbc); % <--- 使用这个版本

% --- 6. 标准 Newton 迭代 ---
fprintf('--- 开始标准 Newton 迭代 (线搜索，使用非线性残差评估) ---\n');
iter = 0; x = zeros(2*Npb+Np, 1); final_residual = NaN; iterations_taken = 0; success = true;

% 计算初始残差 (向量和范数)
fprintf('计算初始残差 (使用非线性残差函数)...\n');
try; res_iter_k = residual_handle(x); res_norm = norm(res_iter_k); % 存储初始残差向量
catch ME_res_init; error('计算初始残差出错: %s.', ME_res_init.message); end
if ~isfinite(res_norm); error('初始残差 NaN/Inf。'); end
res_0_norm = res_norm; fprintf('Iter %d: 初始残差范数 = %.4e\n', iter, res_norm);
tol = max(atol, res_0_norm * rtol); fprintf('收敛容忍度 tol = %.4e\n', tol);
residual_history = [res_norm];

if res_norm <= tol
    fprintf('******************* 初始即收敛 *******************\n');
    success = true; iterations_taken = 0; final_residual = res_norm;
else
    % --- 线搜索参数 ---
    ls_options.alpha_init = 1.0; % Newton 通常从 alpha=1 开始
    ls_options.beta = 0.5; ls_options.c = 1e-4;
    ls_options.max_iter = 10; ls_options.min_alpha = 1e-7;
    fprintf('线搜索参数: alpha_init=%.1f, beta=%.1f, c=%.1e, max_iter=%d, min_alpha=%.1e\n', ...
            ls_options.alpha_init, ls_options.beta, ls_options.c, ls_options.max_iter, ls_options.min_alpha);

    % --- 标准 Newton 迭代循环 ---
    tic_iteration = tic;
    while iter < iter_max && res_norm > tol
        iter = iter + 1;
        x_k = x; % 当前解 u_{k-1}
        v0 = [x_k(1:Npb), x_k(Npb+1:2*Npb)]; % v0 = u_{k-1}

        fprintf('--- Iter %d (当前残差: %.4e) ---\n', iter, res_norm);

        % 1. 组装 An1(v0)
        fprintf('  1. 组装 An1(v0)...\n');
        try; An1 = assemble_An1_v(P, T, Pb, Tb, gauss_bary, weight, p_fem, v0); % 内部转换
        catch ME_an1; warning('Iter %d: 组装 An1 出错: %s', iter, ME_an1.message); success = false; break; end

        % 2. 组装 An2(v0) (标准 Newton)
        fprintf('  2. 组装 An2(v0) (标准)...\n');
        try; An2 = assemble_An2_v_Frechat(P, T, Pb, Tb, gauss_bary, weight, p_fem, v0); % <--- 使用标准 An2 函数, 内部转换
        catch ME_an2; warning('Iter %d: 组装 An2 出错: %s', iter, ME_an2.message); success = false; break; end

        % 3. 构建 Newton 雅可比矩阵 (Jacobian) J(x_k)
        fprintf('  3. 构建 Newton Jacobian 矩阵...\n');
        Jacobian = [As_block + An1 + An2, -B_div'; B_div, sparse(Np,Np)]; % 包含 A0 部分

        % 4. 计算右端项 RHS = -Residual(x_k)
        fprintf('  4. 计算负残差向量 -R(x_k)...\n');
        rhs = -res_iter_k; % 使用上一步/初始的残差向量

        % 5. 应用直接消元法边界条件到 Jacobian 和 RHS
        fprintf('  5. 应用直接消元法边界条件到 Jacobian 和 RHS...\n');
        try; [Jac_bc, rhs_bc] = add_Dirichlet_BC(Jacobian, rhs, Dbc); % <--- 使用直接法
        catch ME_bc; warning('Iter %d: 应用边界条件(直接法)出错: %s', iter, ME_bc.message); success = false; break; end
        % 注意: add_Dirichlet_BC 如何处理 rhs 很重要。标准做法是清零对应行。

        % 6. 求解线性系统 J_bc * delta_x = rhs_bc 得到更新量 delta_x
        fprintf('  6. 求解 Newton 线性系统 J * delta_x = -R(x_k)...\n'); tic_solve = tic;
        try
            delta_x = Jac_bc \ rhs_bc; % 求解更新量 delta_x
            if any(isnan(delta_x)) || any(isinf(delta_x)); warning('Iter %d: Newton 更新量含 NaN/Inf！', iter); success = false; break; end
        catch ME_solve; warning('Iter %d: 求解线性系统出错: %s', iter, ME_solve.message); success = false; break; end
        fprintf('     线性系统求解完成 (%.2f 秒)。\n', toc(tic_solve));
        norm_delta_x = norm(delta_x);
        fprintf('     更新量范数 ||delta_x|| = %.4e\n', norm_delta_x);
        % 可以增加基于更新量大小的收敛判断: if norm_delta_x < some_tol; break; end
        if norm_delta_x < 1e-14 * (1 + norm(x_k)); fprintf('     更新量过小，迭代终止。\n'); break; end

        % 7. 执行线搜索 (寻找最优步长 alpha)
        res_norm_k = res_norm; % 记录当前残差范数
        fprintf('  7. 执行线搜索 (目标: 减小非线性残差范数 %.4e)...\n', res_norm_k); tic_ls = tic;
        try % 包裹线搜索以防内部错误
             % 使用非线性残差函数评估线搜索
            [x_new, alpha, success_ls, res_norm_new] = perform_line_search(x_k, delta_x, res_norm_k, residual_handle, ls_options);
        catch ME_ls; warning('Iter %d: 线搜索函数执行出错: %s', iter, ME_ls.message); success = false; break; end
        fprintf('     线搜索完成 (%.2f 秒)。\n', toc(tic_ls));

        % 8. 检查线搜索结果并更新解和残差
        if ~success_ls
            warning('Iter %d: 线搜索失败！', iter); success = false; break; % 不更新 x 和 res_norm
        end
        x = x_new;                  % 更新解
        res_norm = res_norm_new;    % 更新残差范数
        % 为了下一次迭代的 RHS，需要重新计算残差向量
        try; res_iter_k = residual_handle(x); catch ME_res_next; warning('Iter %d: 计算下一步残差时出错: %s', iter, ME_res_next.message); success = false; break; end % 更新残差向量

        fprintf('     线搜索成功: alpha = %.2e, 新残差范数 = %.4e\n', alpha, res_norm);
        residual_history = [residual_history; res_norm];

        % 9. 检查停滞
        if alpha < ls_options.min_alpha * 1.1; fprintf('     警告: 步长 alpha 小 (%.2e)。\n', alpha); end

    end % --- End Newton while loop ---
    total_iteration_time = toc(tic_iteration);
    fprintf('--- 标准 Newton 迭代循环结束 (总耗时 %.2f 秒) ---\n', total_iteration_time);

    if iter > 0; iterations_taken = iter; final_residual = res_norm; end
end % --- End of "if res_norm > tol" block ---

% --- 12. 输出最终状态和结果 ---
fprintf('\n--- 标准 Newton 迭代总结 ---\n');
if final_residual <= tol && success
    fprintf('[成功] Newton(Std)+LS+DirectBC 求解器在 %d 次迭代后收敛！\n', iterations_taken);
    fprintf('  粘性系数 mu = %.4e\n', mu);
    fprintf('  最终残差范数 = %.4e (容忍度 tol = %.4e)\n', final_residual, tol);
else
    fprintf('[失败] Newton(Std)+LS+DirectBC 求解器未能在 %d 次迭代内收敛或中途失败。\n', iterations_taken);
    fprintf('  粘性系数 mu = %.4e\n', mu);
    fprintf('  最终残差范数 = %.4e (容忍度 tol = %.4e)\n', final_residual, tol);
    if ~success; fprintf('  失败原因：迭代过程中出错或线搜索失败。\n');
    elseif iter == iter_max && final_residual > tol; fprintf('  失败原因：达到最大迭代次数 (%d)。\n', iter_max); end
    warning('求解器未收敛或失败！结果可能不可靠。');
end

% --- 13. 后处理和可视化 (与之前版本相同) ---
if success || iterations_taken > 0
    fprintf('提取并绘制最终结果...\n');
    try; % 包裹绘图代码块
        u_computed=x(1:Npb); v_computed=x(Npb+1:2*Npb); if issparse(u_computed); u_computed=full(u_computed); end; if issparse(v_computed); v_computed=full(v_computed); end
        nodes_inside_holes_p2=false(Npb, 1); for i=1:num_holes; xc=holes(i,1); yc=holes(i,2); r=holes(i,3); try; nodes_inside_holes_p2=nodes_inside_holes_p2 | (dcircle(Pb, xc, yc, r) < -1e-8); catch; break; end; end; u_computed(nodes_inside_holes_p2)=NaN; v_computed(nodes_inside_holes_p2)=NaN;
        fprintf('Generating Delaunay triangulation for velocity...\n'); try; tri_v=delaunay(Pb(:,1), Pb(:,2)); delaunay_ok=true; catch; warning('Delaunay failed for velocity plot'); delaunay_ok=false; end
        if delaunay_ok
            plot_title_base = sprintf('Newton(Std) mu=%.4e, Iter=%d (DirectBC)', mu, iterations_taken); % 更新标题
            figure('Name', sprintf('U Vel (%s - Delaunay)', plot_title_base)); try; trisurf(tri_v, Pb(:,1), Pb(:,2), u_computed); title(sprintf('U Velocity (%s)', plot_title_base)); xlabel('x'); ylabel('y'); colorbar; shading interp; view(2); axis equal; axis([0 L 0 D]); catch ME; warning('Plot U failed: %s',ME.message); end
            figure('Name', sprintf('V Vel (%s - Delaunay)', plot_title_base)); try; trisurf(tri_v, Pb(:,1), Pb(:,2), v_computed); title(sprintf('V Velocity (%s)', plot_title_base)); xlabel('x'); ylabel('y'); colorbar; shading interp; view(2); axis equal; axis([0 L 0 D]); catch ME; warning('Plot V failed: %s',ME.message); end
            figure('Name', sprintf('Vel Mag (%s - Delaunay)', plot_title_base)); try; velocity_magnitude = sqrt(u_computed.^2 + v_computed.^2); trisurf(tri_v, Pb(:,1), Pb(:,2), velocity_magnitude); title(sprintf('Velocity Magnitude |u| (%s)', plot_title_base)); xlabel('x'); ylabel('y'); colorbar; shading interp; view(2); axis equal; axis([0 L 0 D]); catch ME; warning('Plot Mag failed: %s',ME.message); end
        else; fprintf('Skipping Delaunay plots due to triangulation failure.\n'); end
        figure('Name', sprintf('Streamlines (%s)', sprintf('Newton(Std) mu=%.4e, Iter=%d (DirectBC)', mu, iterations_taken))); % 更新标题
        try; [X_reg, Y_reg] = meshgrid(linspace(0, L, 80), linspace(0, D, 50)); interp_u=scatteredInterpolant(Pb(:,1), Pb(:,2), u_computed,'linear','none'); interp_v=scatteredInterpolant(Pb(:,1), Pb(:,2), v_computed,'linear','none'); U_reg=interp_u(X_reg, Y_reg); V_reg=interp_v(X_reg, Y_reg); inside_holes_reg=false(size(X_reg)); for i=1:num_holes; try; points_reg=[X_reg(:), Y_reg(:)]; dist_vals_reg=dcircle(points_reg, holes(i,1), holes(i,2), holes(i,3)); inside_hole_i_reg=reshape(dist_vals_reg < -1e-8, size(X_reg)); inside_holes_reg=inside_holes_reg | inside_hole_i_reg; catch; break; end; end; U_reg(inside_holes_reg)=NaN; V_reg(inside_holes_reg)=NaN; hold on; streamslice(X_reg, Y_reg, U_reg, V_reg, 1.5); plot([0 L L 0 0], [0 0 D D 0], 'k-', 'LineWidth', 1); for i = 1:num_holes; viscircles(holes(i,1:2), holes(i,3), 'Color', 'k', 'LineWidth', 1); end; hold off; title(sprintf('Streamlines (Newton(Std) mu=%.4e)', mu)); xlabel('x'); ylabel('y'); axis equal; axis([0 L 0 D]); catch ME_stream; warning('Plot Streamlines failed: %s', ME_stream.message); end
        fprintf('Identifying boundary nodes for plotting...\n'); try; Dbc1_plot=index_val_Dirichlet_BC_channel(P,T,Pb,Tb,0,D,holes,g1_inlet,g1_wall); Dbc2_plot_nodes=index_val_Dirichlet_BC_channel(P,T,Pb,Tb,0,D,holes,g2_all_dirichlet,g2_all_dirichlet); plot_nodes_u=[];if ~isempty(Dbc1_plot); plot_nodes_u=Dbc1_plot(:,1); end; plot_nodes_v=[]; if ~isempty(Dbc2_plot_nodes); plot_nodes_v=Dbc2_plot_nodes(:,1); end; plot_nodes_all_vel=unique([plot_nodes_u; plot_nodes_v]); plot_nodes_on_holes=[]; if ~isempty(plot_nodes_all_vel); coords_boundary=Pb(plot_nodes_all_vel,:); is_on_a_hole=false(length(plot_nodes_all_vel),1); hole_plot_tol=1e-2; for i=1:length(plot_nodes_all_vel); for h=1:num_holes; try; if abs(dcircle(coords_boundary(i,:), holes(h,1), holes(h,2), holes(h,3))) < hole_plot_tol; is_on_a_hole(i)=true; break; end; catch; break; end; end; end; plot_nodes_on_holes=plot_nodes_all_vel(is_on_a_hole); plot_nodes_outer=setdiff(plot_nodes_all_vel, plot_nodes_on_holes); else; plot_nodes_outer=[]; end; fprintf('Distinguished %d outer & %d hole boundary nodes for plot.\n',length(plot_nodes_outer),length(plot_nodes_on_holes)); figure('Name',sprintf('Identified Boundary Nodes (Newton(Std) mu=%.4e)',mu)); triplot(T, P(:,1), P(:,2),'Color',[0.7 0.7 0.7]); hold on; if ~isempty(plot_nodes_outer); plot(Pb(plot_nodes_outer, 1), Pb(plot_nodes_outer, 2),'b.','MarkerSize',5,'DisplayName','Outer Boundary'); end; if ~isempty(plot_nodes_on_holes); plot(Pb(plot_nodes_on_holes, 1), Pb(plot_nodes_on_holes, 2),'ro','MarkerSize',4,'MarkerFaceColor','r','DisplayName','Hole Boundary'); end; title(sprintf('Identified P2 Boundary Nodes (Red=Holes), mu=%.4e',mu)); axis equal; axis([0 L 0 D]); xlabel('x'); ylabel('y'); legend show; for j=1:num_holes; viscircles(holes(j,1:2), holes(j,3),'Color','k','LineStyle','--'); end; hold off; catch ME_bcplot; warning('Could not plot boundary nodes: %s',ME_bcplot.message); end
    catch ME_main_plot; warning('Error during post-processing or plotting: %s', ME_main_plot.message); end % End plotting try-catch
else; fprintf('Skipping results plotting due to no iterations or failure.\n'); end

% --- 14. 绘制残差历史 ---
if length(residual_history) > 1
    fprintf('Plotting residual history...\n');
    try
        figure('Name', sprintf('Residual History (Newton(Std)+LS+DirectBC, mu=%.4e)', mu)); % 更新标题
        semilogy(0:length(residual_history)-1, residual_history, '-o', 'LineWidth', 1.5, 'MarkerSize', 6);
        xlabel('Iteration'); ylabel('Residual Norm (Nonlinear)'); % 更新标签
        title(sprintf('Convergence History (Newton(Std)+LS+DirectBC, mu=%.4e)', mu)); % 更新标题
        grid on; set(gca, 'YScale', 'log'); hold on;
        plot(get(gca,'XLim'), [tol tol], 'r--', 'LineWidth', 1, 'DisplayName', sprintf('Tol=%.1e', tol));
        legend('Residual', 'Tolerance', 'Location', 'southwest'); hold off; % 更新图例
    catch ME_resplot; warning('Error plotting residual history: %s', ME_resplot.message); end
else; fprintf('Insufficient residual history data (only %d point(s)), skipping plot.\n', length(residual_history)); end

fprintf('\n--- Main function execution finished ---\n');
fprintf('======================================================================================\n');

end % --- End main function ---

% === 需要的子函数占位符 ===
% ... (同前, 确保所有函数存在且正确处理内部坐标转换) ...
% 特别需要: assemble_An2_v (标准定义版本), residual_navier_stokes_nonlinear (标准残差)