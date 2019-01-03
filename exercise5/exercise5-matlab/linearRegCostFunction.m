function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


    % We can use matrix multiplication to computer our sums
    sum1 = (h(theta, X)-y)' * (h(theta, X)-y);

    % Special case: don't regularize theta(1), therefore only use theta(2:end)
    reg1 = theta(2:end)'*theta(2:end);
    
    J = 0.5 * (1/m) * sum1 + 0.5 * (lambda/m) * reg1;

    % Again, the sums for the gradient is computed using a matrix mutliplication
    sum2 = X' * (h(theta, X) - y);
    reg2 = lambda/m * theta;
    % Special case: don't regularize theta(1)
    reg2(1) = 0;
    grad = 1/m * sum2 + reg2;









% =========================================================================

grad = grad(:);

end
