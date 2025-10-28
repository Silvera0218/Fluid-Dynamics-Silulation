function [x_new, alpha_final, ls_success, res_norm_new] = perform_line_search(x_k, delta_x, res_norm_k, residual_func, ls_options)


    % Extract parameters from options structure
    alpha_init = ls_options.alpha_init;
    beta = ls_options.beta;
    c = ls_options.c;
    max_iter = ls_options.max_iter;
    min_alpha = ls_options.min_alpha;

    alpha = alpha_init;
    ls_iter = 0;
    ls_success = false;
    x_new = x_k; % Default to old solution if search fails
    res_norm_new = res_norm_k; % Default residual

    while ls_iter < max_iter && alpha >= min_alpha
        ls_iter = ls_iter + 1;
        x_trial = x_k + alpha * delta_x; % Calculate trial point

        % --- Calculate residual at trial point using the provided function ---
        try
            res_trial_vec = residual_func(x_trial);
            res_norm_trial = norm(res_trial_vec);
        catch ME_res
             warning('LineSearch: Residual function failed at alpha=%.2e: %s. Reducing alpha.', alpha, ME_res.message);
             res_norm_trial = Inf; % Treat function failure as condition not met
        end

        fprintf(' LS Iter %d: alpha=%.2e, ||Res_trial||=%.4e\n', ls_iter, alpha, res_norm_trial);

        % Check Armijo sufficient decrease condition
        sufficient_decrease_rhs = res_norm_k * (1 - c * alpha);
        if res_norm_trial <= sufficient_decrease_rhs
            fprintf('    LS Iter %d: Sufficient decrease condition met (%.4e <= %.4e).\n', ls_iter, res_norm_trial, sufficient_decrease_rhs);
            x_new = x_trial;         % Accept the trial step
            res_norm_new = res_norm_trial; % Update residual norm
            alpha_final = alpha;       % Store accepted alpha
            ls_success = true;
            break; % Exit line search loop
        end

        % Reduce step size if condition not met
        alpha = alpha * beta;

    end % End line search while loop

    if ~ls_success
         warning('Line Search failed to find suitable step size after %d iterations (alpha=%.2e). Using previous solution.', ls_iter, alpha);
         alpha_final = 0; % Indicate failure with alpha=0
    end

    fprintf('  Line Search finished: Success=%d, Final Alpha=%.2e, New Residual=%.4e\n', ls_success, alpha_final, res_norm_new);

end % End of function perform_line_search