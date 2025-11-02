function [Gobs] = construct_observer(G,Ge)
    %waiting states set X' and ready states set X''
    X_prime = strcat(G.X, "'"); 
    X_double_prime = strcat(G.X, "''");
    
    % Create a custom comparison function to compare whether two cell arrays are equal.
    compareFcn = @(x, y) isequal(x, y); 
    
    z_prime_0 = D_epsilon(Ge.q0, Ge); 
    Z_prime_new = {z_prime_0};
    Z_prime = {}; 

    z_0 = zeta(z_prime_0, G);
    Z = {z_0};
    Delta_o = {};

    while ~isempty(Z_prime_new)
        z_prime = Z_prime_new{1};
        for i = 1:length(Ge.Ye)
            y = Ge.Ye{i};

            %function alpha
            alpha = {};
            alpha_1 = {};
            if strcmp(string(y), 'delta') 
                for j = 1:length(z_prime)
                    q = z_prime{j};
                    xx = D_y(q, y, Ge);
                    match = cellfun(@(x) compareFcn(x, xx), alpha);
                    if ~any(match)
                        alpha = [alpha; xx];
                    end
                    %Keep ready global states.
                    if ismember(q{1}, X_double_prime)
                        match = cellfun(@(x) isequal(x, q), alpha_1);
                        if ~any(match)
                            alpha_1 = [alpha_1; {q}];
                        end
                    end
                end

                %Remove duplicate elements
                combined = [alpha; alpha_1];
                unique_elements = {};
                for m = 1:numel(combined)
                    if ~any(cellfun(@(x) compareFcn(x, combined{m}), unique_elements))
                        unique_elements = [unique_elements; combined(m)];
                    end
                end
                alpha = unique_elements;

            else
                for j = 1:length(z_prime)
                    q = z_prime{j};%string
                    xx = D_y(q, y, Ge);

                    %Remove duplicate elements
                    combined = [alpha; xx];
                    unique_elements = {};
                    for m = 1:numel(combined)
                        if ~any(cellfun(@(x) compareFcn(x, combined{m}), unique_elements))
                            unique_elements = [unique_elements; combined(m)];
                        end
                    end
                    alpha = unique_elements;
                end
            end
            
            %function beta
            beta = {};
            for r = 1:length(alpha)
                    qq = alpha{r};
                    xxx = D_epsilon(qq, Ge);
                    
                    %Remove duplicate elements
                    combined = [beta; xxx];
                    unique_elements = {};
                    for m = 1:numel(combined)
                          if ~any(cellfun(@(x) compareFcn(x, combined{m}), unique_elements))
                                unique_elements = [unique_elements; combined(m)];
                          end
                    end
                    beta = unique_elements;
            end

            z_prime_bar = beta;

            %update Z_prime_new
            Z_union = [Z_prime_new; Z_prime];
            if isempty(z_prime_bar)
                isMember = true;
            else
                isMember = any(cellfun(@(x) isequal(x,z_prime_bar), Z_union));
            end
            if ~isMember
                Z_prime_new = [Z_prime_new; {z_prime_bar}];
            end

            %Modify the state of the observer
            z = zeta(z_prime, G); 
            if isempty(z_prime_bar)
                z_bar = {};
            else
                z_bar = zeta(z_prime_bar, G);
            end
            
            %update Z (states set of observer)
            if ~isempty(z_bar)
                   match = cellfun(@(x) compareFcn(x, z_bar), Z);
                   if ~any(match)
                        Z = [Z; {z_bar}];
                   end
            end

            %update Delta_o (transitions of observer)
            if ~isequal(z, z_bar) && ~isempty(z_bar)
                   match = cellfun(@(x) compareFcn(x, [z, string(y), z_bar]), Delta_o);
                   if ~any(match)
                        Delta_o = [Delta_o; {[z, string(y), z_bar]}];%cell(string)
                   end
            end
        end

        Z_prime_new(1) = [];%Remove the z_prime from Z_prime_new 
        Z_prime= [Z_prime; {z_prime}];%update Z_prime
    end
    
    Gobs.Z = Z;
    Gobs.Ye = Ge.Ye;
    Gobs.Delta_o = Delta_o;
    Gobs.z0 = z_0;
end
