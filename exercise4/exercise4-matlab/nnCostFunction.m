function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

    % Part 1: Feedforward the neural network and return the cost in the
    %         variable J. 
    
    % Add additional bias node to X
    X = [ones(m, 1) X];

    % Theta1, Theta2 need to be transposed, since h(theta, X) expects theta to be a column vector
    % In Theta1, Theta2 however, the parameters for each node are represented as a row

    % Hidden layer
    alpha2 = h(Theta1', X);
    
    % Add additional bias node alpha2(0)
    alpha2 = [ones(m, 1) alpha2];
    
    % Output layer
    alpha3 = h(Theta2', alpha2);
    
    % Cost function without regularization term
    y_Vec = zeros(m, num_labels);
    for i =1:m
        y_Vec(i, y(i)) = 1;
    end

    % We can use matrix multiplication to compute our inner sum
    inner = -log(alpha3)*y_Vec' - log(1-alpha3)*(1-y_Vec') ;
    sum1 = sum(diag(inner));
    
    J = 1/m * sum1; 


    % Part 2: Implement the backpropagation algorithm to compute the gradients
    %         Theta1_grad and Theta2_grad. You should return the partial derivatives of
    %         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
    %         Theta2_grad, respectively. 

    % Output layer
    delta3 = alpha3 - y_Vec;
    
    % Hidden layer
    delta2 = delta3 * Theta2 .* alpha2 .* (1-alpha2);
    delta2 = delta2(:, 2:end);
    
    Delta1 = delta2' * X;
    Delta2 = delta3' * alpha2;
    
    Theta1_grad = 1/m * Delta1;
    Theta2_grad = 1/m * Delta2;

    %
    % Part 3: Implement regularization with the cost function and gradients.
    %    
    
    % Regularization term for the cost function
    Theta1_squared = Theta1 .^ 2;
    Theta2_squared = Theta2 .^ 2;
    
    Theta1_squared = Theta1_squared(:, 2:end);
    Theta2_squared = Theta2_squared(:, 2:end);
    
    reg_sum = sum(Theta1_squared(:)) + sum(Theta2_squared(:));
    J = J + lambda /(2*m) * reg_sum;

    % Regularization terms for the gradient matrices
    R1 = Theta1;
    R1(:, 1) = 0;
    R2 = Theta2;
    R2(:, 1) = 0;

    Theta1_grad = Theta1_grad + lambda/m * R1;
    Theta2_grad = Theta2_grad + lambda/m * R2;
    


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
