% Define the SOA system
G.X = {'x0', 'x1', 'x2'}; % Set of states
G.Y = {1,2,3};
G.h = containers.Map({'x0', 'x1', 'x2'}, {[1, 2], [1, 3], [2, 3]}); % Output alphabet (fixed sets for each state)
G.B = {'x0', 'x1'; 'x1', 'x2'; 'x2', 'x1'}; % Set of arcs
G.x0 = 'x0'; % Initial state
G.y0 = 1; % Set initial output symbol to 1

delta = 0.5; % Set delta to 0.5
t = 0:delta:10; % Time interval

% Simulate the SOA system
x = cell(1, length(t)); % State of the system
y = cell(1, length(t)); % Output of the system
x{1} = G.x0;
y{1} = G.y0;
for i = 2:length(t)
    next_states = G.B(strcmp(G.B(:,1), x{i-1}), :); % Possible next states
    r = randi(size(next_states, 2)); % Choose one next state randomly
    x{i} = next_states{1,r};
    current_state = x{i};
    output_symbols = G.h(current_state);
    output_symbol = datasample(output_symbols,1); % Choose one output symbol randomly
    y{i} = output_symbol;
end

% Create the state-output table
state_output_table = table(x', y', 'VariableNames', {'State', 'Output'});

% Plot the state and output of the system
figure;
subplot(2,1,1);
stairs(t, cellfun(@(c) find(strcmp(G.X,c)), x), 'o-'); % Use stairs function instead of plot
xlabel('Time');
ylabel('State');
ylim([0, length(G.X)+1]);
set(gca, 'ytick', 1:length(G.X), 'yticklabel', G.X);
title('State of the SOA system');
subplot(2,1,2);
stairs(t, [y{:}], 'o-'); % Use stairs function instead of plot
xlabel('Time');
ylabel('Output');
ylim([min([G.h('x0'), G.h('x1'), G.h('x2')])-1, max([G.h('x0'), G.h('x1'), G.h('x2')])+1]);
ytick_vals = unique([G.h('x0'), G.h('x1'), G.h('x2')]);
ytick_labels = arrayfun(@num2str, ytick_vals, 'UniformOutput', false);
set(gca, 'ytick', ytick_vals, 'yticklabel', ytick_labels);
title('Output of the SOA system');

% Display the state-output table
figure;
uitable('Data', state_output_table{:,:}, 'ColumnName', state_output_table.Properties.VariableNames, 'RowName', [], 'Units', 'Normalized', 'Position', [0, 0, 1, 1]);

% Call the construct_evolution_automaton function
Ge = construct_evolution_automaton(G);

% Display the evolution automaton
disp(Ge);
