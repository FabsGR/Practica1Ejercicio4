% Paso 1: Cargar los datos
data = csvread('irisbin.csv');
inputs = data(:, 1:4); % Características
targets = data(:, 5:7); % Etiquetas binarias

% Paso 2: Dividir los datos en conjuntos de entrenamiento y generalización
split_ratio = 0.8; % 80% para entrenamiento
num_samples = size(data, 1);
num_train_samples = round(split_ratio * num_samples);

train_inputs = inputs(1:num_train_samples, :);
train_targets = targets(1:num_train_samples, :);
test_inputs = inputs(num_train_samples+1:end, :);
test_targets = targets(num_train_samples+1:end, :);

% Paso 3: Definir la arquitectura de la red neuronal
input_size = 4; % Número de características de entrada
output_size = 3; % Número de clases
hidden_layer_size = [15, 15, 15]; % Número de neuronas en la capa oculta

net = feedforwardnet(hidden_layer_size);
net = configure(net, train_inputs', train_targets');

% Paso 4: Entrenar el modelo
net.trainParam.epochs = 1000; % Número de épocas de entrenamiento
net = train(net, train_inputs', train_targets');

% Paso 5: Evaluar el modelo
y_test = net(test_inputs');

% Paso 6: Redondear las salidas predichas
predicted_test_labels = y_test';
rounded_predicted_test_labels = round(predicted_test_labels);
% Encuentra los valores redondeados a 0 y ajústalos a 1 o -1 manteniendo el signo original
rounded_predicted_test_labels(rounded_predicted_test_labels == 0) = sign(predicted_test_labels(rounded_predicted_test_labels == 0));


% Paso 7: Comparar con las etiquetas reales
real_test_labels = test_targets;

% Paso 8: Mostrar resultados
ytestt = y_test';
% Define el mapeo de etiquetas binarias a clases y nombres
label_mapping = [-1, -1, 1; -1, 1, -1; 1, -1, -1];
class_binary_names = {'[-1, -1, 1] (setosa)', '[-1, 1, -1] (versicolor)', '[1, -1, -1] (virginica)'};

% Obtén las etiquetas mapeadas para los conjuntos de prueba
mapped_test_labels = zeros(size(rounded_predicted_test_labels, 1), 1);

for i = 1:size(rounded_predicted_test_labels, 1)
    for j = 1:size(label_mapping, 1)
        if isequal(rounded_predicted_test_labels(i, :), label_mapping(j, :))
            mapped_test_labels(i) = j;
            break;
        end
    end
end

% Obtén las etiquetas reales mapeadas
mapped_real_test_labels = zeros(size(real_test_labels, 1), 1);

for i = 1:size(real_test_labels, 1)
    for j = 1:size(label_mapping, 1)
        if isequal(real_test_labels(i, :), label_mapping(j, :))
            mapped_real_test_labels(i) = j;
            break;
        end
    end
end

% Muestra los resultados
disp('Resultados en el conjunto de prueba:');
fprintf('Etiquetas reales:\n');
for i = 1:length(real_test_labels)
    fprintf('%s \n', class_binary_names{mapped_real_test_labels(i)});
end

fprintf('Etiquetas predichas (clases):\n');
for i = 1:length(mapped_test_labels)
    fprintf('Binario predicho: [%f,%f,%f] | Redondeando: %s \n',ytestt(i,1),ytestt(i,2),ytestt(i,3), class_binary_names{mapped_test_labels(i)});
end

% Llama a la función de "leave-k-out"
k = 10; 
[average_error_k, std_deviation_k, expected_error_k] = leave_k_out_mlp(train_inputs, train_targets, k);

% Llama a la función de "leave-one-out"
[average_error_loo, std_deviation_loo, expected_error_loo] = leave_one_out_mlp(train_inputs, train_targets);

% Imprime los resultados de ambas validaciones cruzadas
disp(['Leave-k-out - Error esperado: ' num2str(expected_error_k)]);
disp(['Leave-k-out - Error promedio: ' num2str(average_error_k)]);
disp(['Leave-k-out - Desviación estándar: ' num2str(std_deviation_k)]);

disp(['Leave-one-out - Error esperado: ' num2str(expected_error_loo)]);
disp(['Leave-one-out - Error promedio: ' num2str(average_error_loo)]);
disp(['Leave-one-out - Desviación estándar: ' num2str(std_deviation_loo)]);