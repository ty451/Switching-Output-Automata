function [sigma_x] = sigma(x, B)
    % Initialize an empty cell array to store the x' values that meet the condition.
    sigma_x = {};

    % Traverse all cell in B.
    for idx = 1:size(B, 1)
        b = B(idx, :); % Get the current cell in B.
        x_current = b{1}; % The first element x in the cell 
        x_prime = b{2}; % The second element x' in the cell 

        % If the current x is equal to the x value we are looking for
        if strcmp(x_current, x)
            % Add the satisfied x' to the sigma_x set.
            sigma_x = [sigma_x, {x_prime}];
        end
    end
end
