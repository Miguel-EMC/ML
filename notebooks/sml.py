import mxnet as mx
import random
import math
from csv import reader
from math import sqrt
from random import seed
from random import randrange

# Chapter 1 - Data Loading and Preprocessing

def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

def dataset_minmax(dataset):
    if isinstance(dataset, mx.nd.NDArray):
        minmax = mx.nd.stack(dataset.min(axis=0), dataset.max(axis=0), axis=1)
        return minmax
    elif isinstance(dataset, list):
        minmax = list()
        for i in range(len(dataset[0])):
            col_values = [row[i] for row in dataset]
            value_min = min(col_values)
            value_max = max(col_values)
            minmax.append([value_min, value_max])
        return minmax

def normalize_dataset(dataset, minmax):
    if isinstance(dataset, mx.nd.NDArray):
        min_vals = minmax.slice_axis(axis=1, begin=0, end=1).T
        max_vals = minmax.slice_axis(axis=1, begin=1, end=2).T
        dataset[:, :] = (dataset - min_vals) / (max_vals - min_vals)
        return dataset
    elif isinstance(dataset, list):
        for row in dataset:
            for i in range(len(row)):
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
        return dataset

def column_means(dataset):
    if isinstance(dataset, mx.nd.NDArray):
        return mx.nd.mean(dataset, axis=0)
    elif isinstance(dataset, list):
        means = [0 for i in range(len(dataset[0]))]
        for i in range(len(dataset[0])):
            col_values = [row[i] for row in dataset]
            means[i] = sum(col_values) / float(len(dataset))
        return means

def column_stdevs(dataset, means):
    if isinstance(dataset, mx.nd.NDArray):
        return mx.nd.sqrt(((dataset - means) ** 2).sum(axis=0) / (dataset.shape[0] - 1))
    elif isinstance(dataset, list):
        stdevs = [0 for i in range(len(dataset[0]))]
        for i in range(len(dataset[0])):
            variance = [pow(row[i]-means[i], 2) for row in dataset]
            stdevs[i] = sum(variance)
        stdevs = [sqrt(x/(float(len(dataset)-1))) for x in stdevs]
        return stdevs

def standardize_dataset(dataset, means, stdevs):
    if isinstance(dataset, mx.nd.NDArray):
        means = means.reshape((1, -1))
        stdevs = stdevs.reshape((1, -1))
        stdevs = stdevs + 1e-8
        dataset[:, :] = (dataset - means) / stdevs
    elif isinstance(dataset, list):
        for row in dataset:
            for i in range(len(row)):
                row[i] = (row[i] - means[i]) / stdevs[i]

def train_test_split(dataset, split=0.60):
    if isinstance(dataset, mx.nd.NDArray):
        train_size = int(split * dataset.shape[0])
        idx = mx.nd.arange(0, dataset.shape[0])
        idx = mx.nd.shuffle(idx)
        train_idx = idx[:train_size]
        test_idx = idx[train_size:]
        train = mx.nd.take(dataset, train_idx)
        test = mx.nd.take(dataset, test_idx)
        return train, test
    elif isinstance(dataset, list):
        train = list()
        train_size = split * len(dataset)
        dataset_copy = list(dataset)
        while len(train) < train_size:
            index = randrange(len(dataset_copy))
            train.append(dataset_copy.pop(index))
        return train, dataset_copy

def cross_validation_split(dataset, folds=3):
    if isinstance(dataset, mx.nd.NDArray):
        total_rows = dataset.shape[0]
        fold_size = int(total_rows / folds)
        
        # Shuffle indices
        indices = mx.nd.arange(0, total_rows)
        indices = mx.nd.shuffle(indices)
        
        dataset_split = []
        for i in range(folds):
            start = i * fold_size
            end = total_rows if i == folds - 1 else (i + 1) * fold_size
            
            fold_indices = indices[start:end]
            fold = mx.nd.take(dataset, fold_indices)
            dataset_split.append(fold)
        
        return mx.nd.stack(*dataset_split, axis=0)
    elif isinstance(dataset, list):
        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / folds)
        for i in range(folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split


def count_labels(dataset):
    if isinstance(dataset, mx.nd.NDArray):
        labels = dataset[:, -1].astype('int32')
        num_classes = int(labels.max().asscalar()) + 1
        counts = mx.nd.one_hot(labels, num_classes).sum(axis=0)
        
        result = {}
        for i in range(num_classes):
            result[str(i)] = int(counts[i].asscalar())
        
        return result
    else:
        counts = {}
        for row in dataset:
            label = row[-1]
            counts[label] = counts.get(label, 0) + 1
        return counts

def accuracy_metric(actual, predicted):
    if isinstance(actual, mx.nd.NDArray) and isinstance(predicted, mx.nd.NDArray):
        cmp = predicted.astype(actual.dtype) == actual
        correct = cmp.sum().asscalar()
        total = actual.shape[0]
        return (correct / total) * 100.0
    elif isinstance(actual, list) and isinstance(predicted, list):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

def confusion_matrix(actual, predicted):
    if isinstance(actual, mx.nd.NDArray) and isinstance(predicted, mx.nd.NDArray):
        num_classes = int(actual.max().asscalar()) + 1
        actual_one_hot = mx.nd.one_hot(actual.astype('int32'), num_classes)
        predicted_one_hot = mx.nd.one_hot(predicted.astype('int32'), num_classes)
        
        unique = mx.nd.arange(stop=num_classes)
        matrix = mx.nd.dot(actual_one_hot.T, predicted_one_hot)
        
        return unique, matrix
    elif isinstance(actual, list) and isinstance(predicted, list):
        unique = set(actual)
        matrix = [list() for x in range(len(unique))]
        for i in range(len(unique)):
            matrix[i] = [0 for x in range(len(unique))]
        lookup = dict()
        for i, value in enumerate(unique):
            lookup[value] = i
        for i in range(len(actual)):
            x = lookup[actual[i]]
            y = lookup[predicted[i]]
            matrix[y][x] += 1
        return unique, matrix

def print_confusion_matrix(unique, matrix):
    if isinstance(matrix, mx.nd.NDArray):
        print('A/P', ' '.join([str(int(x.asscalar())) for x in unique]))
        for i in range(matrix.shape[0]):
            row_str = ' '.join([str(int(matrix[i, j].asscalar())) for j in range(matrix.shape[1])])
            print(f"{int(unique[i].asscalar())} {row_str}")
    elif isinstance(matrix, list):
        print('(A)' + ' '.join(str(x) for x in unique))
        print('(P)---')
        for i, x in enumerate(unique):
            print("%s| %s" % (x, ' '.join(str(x) for x in matrix[i])))

def mae_metric(actual, predicted):
    if isinstance(actual, mx.nd.NDArray) and isinstance(predicted, mx.nd.NDArray):
        return (mx.nd.abs(actual - predicted).sum() / actual.shape[0]).asscalar()
    elif isinstance(actual, list) and isinstance(predicted, list):
        sum_error = 0.0
        for i in range(len(actual)):
            sum_error += abs(predicted[i] - actual[i])
        return sum_error / float(len(actual))

def rmse_metric(actual, predicted):
    if isinstance(actual, mx.nd.NDArray) and isinstance(predicted, mx.nd.NDArray):
        print(f'RMSE metric shapes — actual: {actual.shape}, predicted: {predicted.shape}')
        
        # Asegura que predicted tenga la misma forma que actual
        if predicted.shape != actual.shape:
            # Si la cantidad de elementos coincide, intenta reshape
            if predicted.size == actual.size:
                predicted = predicted.reshape(actual.shape)
            else:
                # Si no, intenta quitar dimensiones extra
                predicted = predicted.squeeze()
                if predicted.shape != actual.shape:
                    raise ValueError(f"Shape mismatch en RMSE metric: actual {actual.shape} vs predicted {predicted.shape}")
        
        error = predicted - actual
        mse = mx.nd.mean(error ** 2)
        rmse = mx.nd.sqrt(mse)
        return float(rmse.asscalar())
    elif isinstance(actual, list) and isinstance(predicted, list):
        sum_error = 0.0
        for i in range(len(actual)):
            prediction_error = predicted[i] - actual[i]
            sum_error += (prediction_error ** 2)
        mean_error = sum_error / float(len(actual))
        return sqrt(mean_error)

def perf_metrics(actual, y_hat, threshold=0.5):
    if isinstance(actual, mx.nd.NDArray) and isinstance(y_hat, mx.nd.NDArray):
        predicted = y_hat >= threshold
        
        num_classes = int(actual.max().asscalar()) + 1
        actual_one_hot = mx.nd.one_hot(actual.astype('int32'), num_classes)
        predicted_one_hot = mx.nd.one_hot(predicted.astype('int32'), num_classes)
        
        confusion_matrix = mx.nd.dot(actual_one_hot.T, predicted_one_hot)
        
        tp = confusion_matrix[1, 1].asscalar()
        fn = confusion_matrix[1, 0].asscalar()
        fp = confusion_matrix[0, 1].asscalar()
        tn = confusion_matrix[0, 0].asscalar()
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        return fpr, tpr
    elif isinstance(actual, list) and isinstance(y_hat, list):
        tp, fp, tn, fn = 0, 0, 0, 0
        
        for i in range(len(y_hat)):
            if y_hat[i] >= threshold:
                if actual[i] == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if actual[i] == 0:
                    tn += 1
                else:
                    fn += 1
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        return fpr, tpr

def trapz(x, y):
    if isinstance(x, mx.nd.NDArray) and isinstance(y, mx.nd.NDArray):
        sum_val = 0
        for i in range(x.size - 1):
            sum_val += (x[i + 1].asscalar() - x[i].asscalar()) * (y[i].asscalar() + y[i + 1].asscalar()) / 2
        return sum_val
    elif isinstance(x, list) and isinstance(y, list):
        sum_val = 0
        for i in range(len(x) - 1):
            sum_val += (x[i + 1] - x[i]) * (y[i] + y[i + 1]) / 2
        return sum_val

def random_algorithm(train, test):
    if isinstance(train, mx.nd.NDArray) and isinstance(test, mx.nd.NDArray):
        train_labels = train[:, -1]
        labels_list = [int(x.asscalar()) for x in train_labels]
        unique = list(set(labels_list))
        
        num_test_rows = test.shape[0]
        predicted_values = [unique[random.randint(0, len(unique) - 1)] for _ in range(num_test_rows)]
        
        predictions = mx.nd.array(predicted_values)
        return predictions
    else:
        output_values = [row[-1] for row in train]
        unique = list(set(output_values))
        predicted = list()
        for row in test:
            index = randrange(len(unique))
            predicted.append(unique[index])
        return predicted

def zero_rule_algorithm_classification(train, test):
    if isinstance(train, mx.nd.NDArray) and isinstance(test, mx.nd.NDArray):
        label_col = train[:, -1]  # shape (N,)

        labels_list = [int(label_col[i:i+1].asnumpy()[0]) for i in range(int(label_col.shape[0]))]
        
        counter = {}
        for label in labels_list:
            counter[label] = counter.get(label, 0) + 1
        
        prediction_value = max(counter, key=counter.get)
        
        num_test_rows = int(test.shape[0])
        predictions = mx.nd.full((num_test_rows,), prediction_value)
        return predictions
    else:
        output_values = [row[-1] for row in train]
        prediction = max(set(output_values), key=output_values.count)
        predicted = [prediction for i in range(len(test))]
        return predicted

def zero_rule_algorithm_regression(train, test):
    if isinstance(train, mx.nd.NDArray) and isinstance(test, mx.nd.NDArray):
        output_values_nd = train[:, -1]
        sum_nd = output_values_nd.sum()
        count = output_values_nd.shape[0]
        average_nd = sum_nd / count
        prediction_value = average_nd.asscalar()
        
        return mx.nd.full((test.shape[0],), prediction_value)
    elif isinstance(train, list) and isinstance(test, list):
        output_values = [row[-1] for row in train]
        prediction = sum(output_values) / float(len(output_values))
        predicted = [prediction for i in range(len(test))]
        return predicted
    
import mxnet as mx

def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = []

    for i in range(n_folds):
        test = folds[i]
        train = mx.nd.concat(*[folds[j] for j in range(n_folds) if j != i], dim=0)
        
        test_input = test[:, :-1]
        actual = test[:, -1]
        print(f"Fold {i+1} - Test input shape: {test.shape}, Actual shape: {actual.shape}")
        predicted = algorithm(train, test, *args)
        error = predicted - actual
        rmse = rmse_metric(actual, predicted)
        scores.append(rmse)

    return scores

# Neural Network Functions with Tensors

def activate(weights, inputs):
    """Calculate neuron activation using tensors"""
    if isinstance(weights, mx.nd.NDArray) and isinstance(inputs, mx.nd.NDArray):
        bias = weights[-1:]
        weight_vec = weights[:-1]
        return mx.nd.dot(weight_vec, inputs) + bias
    else:
        weights_nd = mx.nd.array(weights) if not isinstance(weights, mx.nd.NDArray) else weights
        inputs_nd = mx.nd.array(inputs) if not isinstance(inputs, mx.nd.NDArray) else inputs
        bias = weights_nd[-1:]
        weight_vec = weights_nd[:-1]
        return mx.nd.dot(weight_vec, inputs_nd) + bias

def transfer(activation):
    """Sigmoid activation function using tensors"""
    if not isinstance(activation, mx.nd.NDArray):
        activation = mx.nd.array([activation])
    return mx.nd.sigmoid(activation)

def transfer_derivative(output):
    """Calculate derivative of sigmoid using tensors"""
    if not isinstance(output, mx.nd.NDArray):
        output = mx.nd.array([output])
    return output * (1.0 - output)

def initialize_network(n_inputs, n_hidden, n_outputs):
    """Initialize neural network with tensor weights"""
    network = []
    
    # Hidden layer with tensor weights
    hidden_weights = mx.nd.random.uniform(-1, 1, shape=(n_hidden, n_inputs + 1))
    hidden_layer = {
        'weights': hidden_weights,
        'outputs': mx.nd.zeros(n_hidden),
        'deltas': mx.nd.zeros(n_hidden)
    }
    network.append(hidden_layer)
    
    # Output layer with tensor weights
    output_weights = mx.nd.random.uniform(-1, 1, shape=(n_outputs, n_hidden + 1))
    output_layer = {
        'weights': output_weights,
        'outputs': mx.nd.zeros(n_outputs),
        'deltas': mx.nd.zeros(n_outputs)
    }
    network.append(output_layer)
    
    return network

def forward_propagate(network, row):
    """Forward propagation using tensor operations"""
    # Convert input to tensor (exclude last element which is the target)
    inputs = mx.nd.array(row[:-1])
    
    for layer in network:
        weights = layer['weights']
        # Separate bias and weight matrix
        bias = weights[:, -1]
        weight_matrix = weights[:, :-1]
        
        # Calculate activation and apply sigmoid
        activation = mx.nd.dot(weight_matrix, inputs) + bias
        layer['outputs'] = mx.nd.sigmoid(activation)
        inputs = layer['outputs']
    
    return network[-1]['outputs']

def backward_propagate_error(network, expected):
    """Backward propagation using tensor operations"""
    # Convert expected to tensor if needed
    if not isinstance(expected, mx.nd.NDArray):
        expected = mx.nd.array(expected)
    
    # Iterate through layers in reverse
    for i in range(len(network) - 1, -1, -1):
        layer = network[i]
        
        if i != len(network) - 1:  # Hidden layer
            next_layer = network[i + 1]
            next_weights = next_layer['weights'][:, :-1]  # Exclude bias
            next_deltas = next_layer['deltas']
            errors = mx.nd.dot(next_weights.T, next_deltas)
        else:  # Output layer
            errors = expected - layer['outputs']
        
        # Calculate deltas using tensor operations
        outputs = layer['outputs']
        layer['deltas'] = errors * outputs * (1.0 - outputs)

def update_weights(network, row, l_rate):
    """Update network weights using tensor operations"""
    # Convert input to tensor
    inputs = mx.nd.array(row[:-1])
    
    for i, layer in enumerate(network):
        # Get current layer inputs
        if i == 0:
            current_inputs = inputs
        else:
            current_inputs = network[i-1]['outputs']
        
        # Add bias term
        input_with_bias = mx.nd.concat(current_inputs, mx.nd.ones(1), dim=0)
        
        # Calculate weight updates using outer product
        deltas = layer['deltas']
        weight_update = mx.nd.outer(deltas, input_with_bias) * l_rate
        
        # Update weights
        layer['weights'] = layer['weights'] + weight_update

def train_network(network, train, l_rate, n_epoch, n_outputs, test=None):
    """Train neural network using tensor operations"""
    train_losses = []
    test_losses = []
    
    for epoch in range(1, n_epoch + 1):
        sum_error = 0.0
        
        for row in train:
            # Forward propagation
            outputs = forward_propagate(network, row)
            
            # Create one-hot encoded expected output
            expected = mx.nd.zeros(n_outputs)
            expected[int(row[-1])] = 1
            
            # Calculate error
            error = expected - outputs
            sum_error += float((error ** 2).sum().asscalar())
            
            # Backward propagation
            backward_propagate_error(network, expected)
            
            # Update weights
            update_weights(network, row, l_rate)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f" >epoch={epoch}, lrate={l_rate:.3f}, error={sum_error:.3f}")
        
        train_losses.append(sum_error)
        
        # Test error if test data provided
        if test is not None:
            test_error = 0.0
            for row in test:
                outputs = forward_propagate(network, row)
                expected = mx.nd.zeros(n_outputs)
                expected[int(row[-1])] = 1
                error = expected - outputs
                test_error += float((error ** 2).sum().asscalar())
            test_losses.append(test_error)
    
    return network, train_losses, test_losses

def predict_nn(network, row):
    """Make prediction using tensor operations"""
    outputs = forward_propagate(network, row)
    return int(outputs.argmax().asscalar())

def back_propagation(train, test, l_rate, n_epoch, n_hidden):
    """Backpropagation algorithm using tensors"""
    n_inputs = len(train[0]) - 1
    n_outputs = len(set(row[-1] for row in train))
    
    # Initialize network with tensors
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    
    # Train network
    trained_network, train_losses, test_losses = train_network(
        network, train, l_rate, n_epoch, n_outputs, test
    )
    
    # Make predictions
    predictions = []
    for row in test:
        prediction = predict_nn(trained_network, row)
        predictions.append(prediction)
    
    return predictions, train_losses, test_losses