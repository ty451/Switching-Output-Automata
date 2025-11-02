function [D_y_q] = D_y(q, y, Ge)
    % Initialize an empty set.
    D_y_q = {};

    % Find all y transitions from q in Ge.Delta.
    y_transitions = cellfun(@(x) isequal(x, q), Ge.Delta(:,1)) & cellfun(@(x) isequal(x, string(y)), Ge.Delta(:,2));

    % Find all states that can be reached through the transition of y.
    reachable_states = Ge.Delta(y_transitions, 3);

    % Add the newly found state to D_y_q, but avoid duplicate additions.
    for state = reachable_states'
        if ~any(cellfun(@(x) isequal(x, state{1}), D_y_q))
            D_y_q = [D_y_q; state];
        end
    end
end
