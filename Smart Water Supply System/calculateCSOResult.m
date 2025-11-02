function flattenedResults = calculateCSOResult(G)
    % Get states and corresponding arrays
    states = keys(G.h);
    arrays = values(G.h);
    aaa =0;
    % Initialize the result cell array, pre-allocating size for efficiency
    results = cell(1, length(states));

    % Use parfor loop to iterate over all states
    parfor i = 1:length(states)
        localG = G;  % Create a copy of G for each iteration
        localG.x0 = states{i};  % Set the current state
        array = arrays{i};  % Get the array corresponding to the current state

        tempResults = cell(1, length(array)); % Initialize a temporary results array

        % Inner loop, iterate over each element in the array for the current state
        for j = 1:length(array)
            localG.y0 = array(j);  % Set the current element from the array
            Ge = construct_evolution_automaton(localG);
            Gobs = construct_observer(localG, Ge);
            [CSO, unopacue_array] = verify_CSO(localG, Gobs);
            % If the CSO condition is not satisfied, save the result
            if ~CSO
                tempResults{j} = {localG.x0, localG.y0, CSO, unopacue_array};
            end
        end

        % Filter out empty elements from the temporary results
        tempResults = tempResults(~cellfun(@isempty, tempResults));
        
        % Save the temporary results into the corresponding slot in the global results
        results{i} = vertcat(tempResults{:});
    end

     % After all parallel iterations end, flatten the nested results cell array
    flattenedResults = vertcat(results{:});
end
