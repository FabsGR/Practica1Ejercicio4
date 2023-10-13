% Definición de la función leave_k_out_mlp que realiza validación cruzada "leave-k-out"
% en una red neuronal de perceptrón multicapa (MLP).
function [average_error, std_deviation, expected_error] = leave_k_out_mlp(train_inputs, train_targets, k)
    % Obtener el número de muestras en el conjunto de entrenamiento.
    num_samples = size(train_inputs, 1);
    
    % Inicializar un vector para almacenar las tasas de error.
    error_rates = zeros(k, 1);
    
    % Inicializar la suma total de errores.
    error_sum = 0;

    % Iniciar el bucle para "leave-k-out".
    for i = 1:k
        % Seleccionar aleatoriamente k índices para el conjunto de prueba.
        test_indices = randperm(num_samples, k);
        
        % Obtener los índices restantes como el conjunto de entrenamiento.
        train_indices = setdiff(1:num_samples, test_indices);

        % Dividir los datos y etiquetas en conjuntos de entrenamiento y prueba.
        train_data = train_inputs(train_indices, :);
        train_labels = train_targets(train_indices, :);
        test_data = train_inputs(test_indices, :);
        test_labels = train_targets(test_indices, :);

        % Definir y configurar la red neuronal MLP.
        input_size = size(train_data, 2);
        output_size = size(train_labels, 2);
        hidden_layer_size = [10, 10, 10];
        net = feedforwardnet(hidden_layer_size);
        net = configure(net, train_data', train_labels');

        % Entrenar la red neuronal.
        net.trainParam.epochs = 1000; % Número de épocas de entrenamiento
        net = train(net, train_data', train_labels');

        % Evaluar el rendimiento en el conjunto de prueba.
        y_testk = net(test_data');

        % Redondear las salidas predichas.
        rounded_predicted_testk_labels = round(y_testk');

        % Calcular la tasa de error.
        error_rate = sum(~isequal(rounded_predicted_testk_labels, test_labels)) / length(test_labels);
        error_rates(i) = error_rate;
        error_sum = error_sum + error_rate;
    end

    % Calcular el error promedio, desviación estándar y error esperado.
    expected_error = error_sum / k;
    average_error = mean(error_rates);
    std_deviation = std(error_rates);
end
