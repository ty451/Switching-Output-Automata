function [zeta_state] = zeta(Q_sub, G)

        X_prime = strcat(G.X, "'");
        X_double_prime = strcat(G.X, "''");
    
        % Extract x_e of all states in Q_sub.
        X_s_Q_sub = cellfun(@(q) q{1}, Q_sub, 'UniformOutput', false);
    
        % Create a new set of states to store the simplified state.
        simplified_states = X_s_Q_sub;
    
        % Traverse all states in the set of states.
        for i = 1:length(X_s_Q_sub)
            x_e = X_s_Q_sub{i}; % Get current status.
    
            % Check if there exists another state, denoted as x'', and x and x' correspond to the same state.
            for j = 1:length(X_s_Q_sub)
                x_e_bar = X_s_Q_sub{j}; % Get another state.
                if all(ismember(x_e, X_prime)) && all(ismember(x_e_bar, X_double_prime))
                    if strcmp(strrep(x_e, "'", ""), strrep(x_e_bar, "''", ""))
                    % If it exists, then delete state x_e from simplified_states.
                    simplified_states = setdiff(simplified_states, {x_e});
                    end
                end
            end
        end
    
        % Return to simplified state.
        sortedsimplified_states = sort(simplified_states);
        str = strjoin(sortedsimplified_states, ', ');
        simplified_states_new = ['{' str '}'];
        
        zeta_state = [string(simplified_states_new), string(Q_sub{1}{2})];
   
end
