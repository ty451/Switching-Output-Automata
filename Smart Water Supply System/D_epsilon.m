function [D_epsilon_q] = D_epsilon(q,Ge)
    % Initialize a collection, first add q itself to it.

    D_epsilon_q = {q};

    % Initialize a queue that needs to be processed, and add q to it.
    queue = {q};

    while ~isempty(queue)
        % Take out a state from the queue.
        current_state = queue{1};

        % Find all epsilon transitions from current_state in Ge.Delta.
        epsilon_transitions = cellfun(@(x) isequal(x, current_state), Ge.Delta(:,1)) & strcmp(Ge.Delta(:,2), 'epsilon');

        % Find all states that can be reached through epsilon transitions.
        reachable_states = Ge.Delta(epsilon_transitions, 3);

        % Add the newly found state to D_epsilon_q and queue, but avoid duplicate additions.
        for state = reachable_states'
            if ~any(cellfun(@(x) isequal(x, state{1}), D_epsilon_q))
                D_epsilon_q = [D_epsilon_q; state];
                queue = [queue; state];
            end
        end

        % Remove the processed status from the queue.
        queue(1) = [];
    end
%end