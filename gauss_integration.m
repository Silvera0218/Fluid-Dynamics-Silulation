function [gauss, weight] = gauss_integration(p)
% Provides Gaussian quadrature points (barycentric coordinates) and weights
% for numerical integration over the standard reference triangle 
% (vertices at (0,0), (1,0), (0,1), Area = 1/2).
% INPUT:
%   p: The maximum polynomial degree to be integrated exactly.
% OUTPUT:
%   gauss: n x 3 matrix of barycentric coordinates [lambda1, lambda2, lambda3].
%   weight: 1 x n vector of weights. The weights are normalized such that 
%           integral(f) approx= sum(weight(i) * f(gauss(i,:))).
%           sum(weight) == 1/2 (Area of reference triangle).

  if p == 1 % p=1, n=1 (Midpoint rule, exact for degree 1)
    gauss = zeros(1, 3); weight = zeros(1, 1);
    gauss(1, :) = [1/3, 1/3, 1/3];
    weight(1) = 1/2;
    
  elseif p == 2 % p=2, n=3 (Edge midpoint rule, exact for degree 2)
    gauss = zeros(3, 3); weight = zeros(1, 3);
    gauss(1, :) = [1/2, 1/2, 0];
    gauss(2, :) = [1/2, 0, 1/2];
    gauss(3, :) = [0, 1/2, 1/2];
    weight(1:3) = 1/6; % 3 * (1/6) = 1/2
    
  elseif p == 3 % p=3, n=4 (Strang 4-point rule, exact for degree 3)
    gauss = zeros(4, 3); weight = zeros(1, 4);
    gauss(1, :) = [1/3, 1/3, 1/3]; 
    gauss(2, :) = [3/5, 1/5, 1/5]; % Using rational representation: [0.6, 0.2, 0.2]
    gauss(3, :) = [1/5, 3/5, 1/5]; % Using rational representation: [0.2, 0.6, 0.2]
    gauss(4, :) = [1/5, 1/5, 3/5]; % Using rational representation: [0.2, 0.2, 0.6]
    weight(1) = -9/32; 
    weight(2:4) = 25/96; % (-9/32) + 3*(25/96) = (-27 + 75)/96 = 48/96 = 1/2
    
  elseif p == 4 % p=4, n=6 (Dunavant 6-point rule, exact for degree 4)
    % References: Dunavant, D. A. (1985). High degree efficient symmetrical 
    %             Gaussian quadrature rules for the triangle. 
    %             International journal for numerical methods in engineering, 
    %             21(6), 1129-1148. (Rule 6-2 / Order 6)
    gauss = zeros(6, 3); 
    weight = zeros(1, 6);
    
    % Define constants for points
    r1 = 0.1012865073235565; % == s2
    s1 = 0.44935674633822175; 
    r2 = 0.7974269853530873;
    s2 = 0.10128650732345635; % == r1
    
    % Define constants for weights (normalized to sum to 1/2)
    w1 = 0.1259391805448271; % Weight for points 1-3
    w2 = 0.0407274861218395; % Weight for points 4-6
                               % Check: 3*w1 + 3*w2 = 0.5
                               
    % Points (Barycentric Coordinates)
    gauss(1, :) = [r1, s1, s1];
    gauss(2, :) = [s1, r1, s1];
    gauss(3, :) = [s1, s1, r1];
    
    gauss(4, :) = [r2, s2, s2];
    gauss(5, :) = [s2, r2, s2];
    gauss(6, :) = [s2, s2, r2];
    
    % Weights
    weight(1:3) = w1;
    weight(4:6) = w2;
    
  else
    error('gauss_integration: Unsupported precision order p = %d requested.', p);
  end
  
end