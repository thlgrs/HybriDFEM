% STEP 1: Apri il file FIG
fig = openfig('S_1_12.fig');  % assicurati che il file .fig sia nella cartella corrente

% STEP 2: Trova tutti gli oggetti 'line' nel grafico
lines = findall(fig, 'Type', 'line');

% STEP 3: I più recenti vengono in alto: inverto per avere l'ordine corretto
lines = flipud(lines);

% STEP 4: Per ogni curva, estraggo i dati e li salvo in un file .csv
for i = 1:min(3, length(lines))
    x = get(lines(i), 'XData');
    y = get(lines(i), 'YData');
    
    % Creo un nome file: es. 'curve_1.csv'
    filename = sprintf('curve_%d_12.csv', i);
    
    % Salvo i dati [X Y] come CSV
    writematrix([x(:), y(:)], filename);
    
    fprintf('✅ Curve %d salvata come "%s"\n', i, filename);
end