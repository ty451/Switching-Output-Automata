function [CSO, unopacue_array] = verify_CSO(G, Gobs)

X_double_prime = strcat(G.X, "''");
opacity_array = {};
A = {};

% Iterate through each cell in Z
for ss = 1:length(Gobs.Z)
    % Get the first element of the cell
    a = Gobs.Z{ss}(1);
    
    % Remove the curly braces from the string
    a = erase(a, "{");
    a = erase(a, "}");

    % Use strsplit to split the string, obtaining an array of elements
    split_a = strsplit(a, ', ');

    % Convert the array of strings to a cell array using cellstr
    restored_sortedsimplified_states = cellstr(split_a);
    result = ismember(restored_sortedsimplified_states, X_double_prime);
    
    if all(result)
        stable = 1;
    else
        stable = 0;
    end
    
    if stable == 1
        % Use erase to remove '' from each element
        restored_sortedsimplified_states = cellfun(@(x) erase(x, "''"), restored_sortedsimplified_states, 'UniformOutput', false);
    
        % Convert the cell array to an array of strings
        string_array = string(restored_sortedsimplified_states);
        is_subset = all(ismember(string_array, G.S));
        % Check if it is a subset of S
        if is_subset
            opacity = false;
            A = [A; {ss}];
        else
            opacity = true;
        end
    
        % Add opacity to the cell array
        opacity_array = [opacity_array; {opacity}];
    end
end

% Use any to check if there is any false in opacity_array
if any(cell2mat(opacity_array) == false)
    CSO = false;
else
    CSO = true;
end

% Create a new cell array
unopacue_array = cell(length(A), 1);

% Extract elements from Gobs.Z and add them to the new cell array
for i = 1:length(A)
    unopacue_array{i} = Gobs.Z{A{i}};
end

end
