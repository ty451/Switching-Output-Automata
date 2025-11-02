function write_data_to_files(Ge,Gobs)

    fid = fopen('Ge_Q.txt', 'w');

    
    if fid == -1
      error('Cannot open file for writing.');
    end

    fprintf(fid, 'Ge.Q:\n');
    for i = 1:length(Ge.Q)
      fprintf(fid, '%s %s\n', Ge.Q{i}(1), Ge.Q{i}(2));
    end

    fclose(fid);
    
    %%
    
    fid = fopen('Gobs_Z.txt', 'w');

    if fid == -1
      error('Cannot open file for writing.');
    end

    fprintf(fid, 'Gobs.Z:\n');
    for i = 1:length(Gobs.Z)
      fprintf(fid, '%s %s\n', Gobs.Z{i}(1), Gobs.Z{i}(2));
    end

    fclose(fid);

    %%
    nRows = size(Ge.Delta, 1); 
    combinedStrings = cell(nRows, 1); 

    for i = 1:nRows
        combinedStrings{i} = [strjoin(Ge.Delta{i, 1}, ' '), ' ', Ge.Delta{i, 2}, ' ', strjoin(Ge.Delta{i, 3}, ' ')];
    end
    T = cell2table(combinedStrings, 'VariableNames', {'CombinedData'});

    writetable(T, 'Ge_Delta.xlsx');

    %%
    new_cell_array = cell(size(Gobs.Delta_o));

    for i = 1:numel(Gobs.Delta_o)
        new_cell_array{i} = strjoin(Gobs.Delta_o{i}, ' ');
    end

    T = cell2table(new_cell_array);
    
    writetable(T, 'Gobs_Delta_o.xlsx');

end
