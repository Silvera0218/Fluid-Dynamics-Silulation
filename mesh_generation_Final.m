function [p,t] = mesh_GEN()  
    % Mesh generation for rectangular domain with circular holes  
    % Quasi-uniform mesh with size h ∈ [0.1, 0.2]  
    
    % Domain parameters  
    L = 17;  % Length of domain  
    D = 10;  % Height of domain  
    
    % Holes: [x, y, radius]  
    holes = [  
        1.5, 5, 1;   % 第一个圆心位置修改为(1.5, 5)  
        5, 7, 1;     % 第二个圆心  
        5, 3, 1;     % 第三个圆心  
    ];  
    
    % Distance function  
    fd = @(p) ddiff(drectangle(p, 0, L, 0, D), ...  
        dcircle(p, holes(1,1), holes(1,2), 1));  
    fd = @(p) ddiff(fd(p), dcircle(p, holes(2,1), holes(2,2), 1));  
    fd = @(p) ddiff(fd(p), dcircle(p, holes(3,1), holes(3,2), 1));  
    
    % Mesh size function (quasi-uniform with variation)  
    function h = mesh_size(p)  
        % Base mesh size with slight variation  
        h_min = 0.1;  
        h_max = 0.2;  
        
        % Create a slightly non-uniform distribution  
        h = h_min + (h_max - h_min) * (1 + sin(p(:,1)/L*2*pi) * 0.2);  
        
        % Ensure mesh size constraints  
        h = max(h_min, min(h_max, h));  
    end  

    % Fixed boundary points  
    pfix = [  
        0, 0;   % Bottom left  
        0, D;   % Top left  
        L, 0;   % Bottom right  
        L, D;   % Top right  
    ];  
    
    % Generate mesh  
    try  
        % Use custom mesh size function  
        [p, t] = distmesh2d(fd, @mesh_size, 0.15, [0, 0; L, D], pfix);  
        
        % Validate mesh size  
        edge_lengths = sqrt(sum((p(t(:,1),:) - p(t(:,2),:)).^2, 2));  
        fprintf('Mesh Size Statistics:\n');  
        fprintf('  Min edge length: %.4f\n', min(edge_lengths));  
        fprintf('  Max edge length: %.4f\n', max(edge_lengths));  
        fprintf('  Mean edge length: %.4f\n', mean(edge_lengths));  
        
        % Visualization  
        figure;  
        triplot(t, p(:,1), p(:,2), 'k-');  
        hold on;  
        plot(pfix(:,1), pfix(:,2), 'ro', 'MarkerFaceColor', 'r');  
        plot(holes(:,1), holes(:,2), 'gx', 'MarkerSize', 10);  
        title('Quasi-Uniform Mesh Generation');  
        xlabel('X');  
        ylabel('Y');  
        axis equal;  
        
        % Save mesh  
        save('domain_mesh2.mat', 'p', 't', 'pfix', 'holes', 'L', 'D');  
        
        fprintf('Mesh generation successful!\n');  
    catch ME  
        fprintf('Mesh generation error: %s\n', ME.message);  
        p = [];  
        t = [];  
    end  
end