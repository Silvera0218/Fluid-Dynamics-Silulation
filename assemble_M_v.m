
function M_scalar = assemble_M_v(Pb, Tb, gauss_bary_input, gauss_weights, p, coef)
  % --- Get Dimensions ---
  Npb = size(Pb, 1);      % Total number of P2 nodes
  [Ne, Nlp] = size(Tb);   % Ne = num elements, Nlp = 6 for P2
  Ng = size(gauss_bary_input, 1); % Number of Gauss points

  % --- Convert Barycentric to Reference Cartesian and Precompute ---
  gauss_ref_cart = gauss_bary_input(:, [2, 3]); % Convert [l1,l2,l3] to [xi, eta] = [l2, l3]
  
  phi_ref = zeros(Nlp, Ng);
  dphi_dxi_ref = zeros(Nlp, Ng);
  dphi_deta_ref = zeros(Nlp, Ng);
  for k = 1:Ng
      % Now use the converted gauss_ref_cart for basis_function calls
      phi_ref(:, k) = basis_function(p, 0, 0, gauss_ref_cart(k, :)); 
      dphi_dxi_ref(:, k) = basis_function(p, 1, 0, gauss_ref_cart(k, :)); 
      dphi_deta_ref(:, k) = basis_function(p, 0, 1, gauss_ref_cart(k, :)); 
  end
  % --- ---

  % --- Initialize sparse matrix components (COO format) ---
  max_entries = Ne * Nlp * Nlp; 
  II = zeros(max_entries, 1);
  JJ = zeros(max_entries, 1);
  SS = zeros(max_entries, 1);
  entry_count = 0;

  % --- Element Loop ---
  for ne = 1:Ne
      element_nodes = Tb(ne, :); 
      if any(element_nodes <= 0) || any(element_nodes > Npb)
          warning('Element %d contains invalid node indices. Skipping.', ne);
          continue;
      end
      element_coords = Pb(element_nodes, :); 

      M_local = zeros(Nlp, Nlp); 

      % --- Gauss Point Loop ---
      for k = 1:Ng
          % Get precomputed values/derivatives at this Gauss point k
          phi_at_k = phi_ref(:, k);         
          dphi_dxi_at_k = dphi_dxi_ref(:, k); 
          dphi_deta_at_k = dphi_deta_ref(:, k); 

          % Calculate Jacobian matrix J and determinant |detJ| at point k
          J_at_k = [dphi_dxi_at_k' * element_coords(:, 1), dphi_deta_at_k' * element_coords(:, 1); 
                    dphi_dxi_at_k' * element_coords(:, 2), dphi_deta_at_k' * element_coords(:, 2)];
          detJ_at_k = det(J_at_k);

          if detJ_at_k <= 1e-12 
              warning('Element %d has near-zero Jacobian determinant (%.2e) at Gauss point %d. Skipping point.', ne, detJ_at_k, k);
              continue; 
          elseif detJ_at_k < 0
               warning('Element %d has negative Jacobian determinant (%.2e) at Gauss point %d. Check node ordering or mesh quality.', ne, detJ_at_k, k);
          end
          abs_detJ_at_k = abs(detJ_at_k);
          
          % Calculate contribution to local mass matrix
          integrand_matrix = (phi_at_k * phi_at_k'); 
          M_local = M_local + integrand_matrix * abs_detJ_at_k * gauss_weights(k);

      end % --- End Gauss Point Loop ---

      % Add local matrix contributions to global COO lists
      for i = 1:Nlp
          for j = 1:Nlp
              if abs(M_local(i, j)) > 1e-15 
                  entry_count = entry_count + 1;
                  if entry_count > max_entries % Resize check
                      new_size = floor(max_entries * 1.2) + Ne * Nlp; 
                      II = [II; zeros(new_size - max_entries, 1)];
                      JJ = [JJ; zeros(new_size - max_entries, 1)];
                      SS = [SS; zeros(new_size - max_entries, 1)];
                      max_entries = new_size;
                      warning('Resized sparse storage during assembly in assemble_M_v.');
                  end
                  II(entry_count) = element_nodes(i);
                  JJ(entry_count) = element_nodes(j);
                  SS(entry_count) = M_local(i, j);
              end
          end
      end
  end % --- End Element Loop ---

  % Assemble the global sparse matrix
  if entry_count > 0
      M_scalar = sparse(II(1:entry_count), JJ(1:entry_count), SS(1:entry_count), Npb, Npb);
  else
      M_scalar = sparse(Npb, Npb); 
  end

  % Apply the overall coefficient
  M_scalar = coef * M_scalar;

end