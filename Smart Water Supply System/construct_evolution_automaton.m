function [Ge] = construct_evolution_automaton(G)
    X = G.X;
    Y = G.Y;
    B = G.B;
    h = G.h;
    x0 = G.x0;
    y0 = G.y0;
    
    %waiting states set X' and ready states set X''
    X_prime = strcat(X, "'");
    X_double_prime = strcat(X, "''");
    Xe = [X_prime, X_double_prime];
    Ye = [Y, {'delta'}];

    q0=[strcat(x0, "'"), num2str(y0)];
    Qnew = {q0};
    Q = {};
    Delta = {};

    while ~isempty(Qnew)
        q = Qnew{1};
        x_e = q(1:end-1);
        y = str2double(q(end));

        if ismember(x_e, X_prime)
            x_e_bar = strrep(x_e, "'", "''");
            q_bar = [x_e_bar, num2str(y)];
            Delta = [Delta; {q, 'delta', q_bar}];
            
            Q_union = [Q; Qnew];
            isMember = any(cellfun(@(x) isequal(x, q_bar), Q_union));
            if ~isMember
                Qnew = [Qnew; {q_bar}];
            end
        elseif ismember(x_e, X_double_prime)
            x = strrep(x_e, "''", "");
            for x_bar = sigma(x, B)
                if ismember(y, h(x_bar{1}))
                    x_e_bar = strcat(x_bar, "'");
                    q_bar = [x_e_bar, num2str(y)];
                    Delta = [Delta; {q, 'epsilon', q_bar}]; %The output remains the same, but the state changes.

                    Q_union = [Q; Qnew];
                    isMember = any(cellfun(@(x) isequal(x, q_bar), Q_union));
                    if ~isMember
                        Qnew = [Qnew; {q_bar}];
                    end
                end
                for y_bar = setdiff(h(x_bar{1}), y)
                    x_e_bar = strcat(x_bar, "'");
                    q_bar = [x_e_bar, num2str(y_bar)];
                    Delta = [Delta; {q, num2str(y_bar), q_bar}]; %Both the state and output have changed.

                    Q_union = [Q; Qnew];
                    isMember = any(cellfun(@(x) isequal(x, q_bar), Q_union));
                    if ~isMember
                        Qnew = [Qnew; {q_bar}];
                    end
                end
            end
            for y_bar = setdiff(h(x), y)
                x_e_bar = strcat(x, "'");
                q_bar = [x_e_bar, num2str(y_bar)];
                Delta = [Delta; {q, num2str(y_bar), q_bar}];%The output changes, but the state remains unchanged.

                 Q_union = [Q; Qnew];
                 isMember = any(cellfun(@(x) isequal(x, q_bar), Q_union));
                 if ~isMember
                       Qnew = [Qnew; {q_bar}];
                 end
            end
        end

       Qnew(1,:) = [];
       Q = [Q; {q}];
    end

    Ge.Q = Q;
    Ge.Ye = Ye;
    Ge.Delta = Delta;
    Ge.q0 = q0;
end



