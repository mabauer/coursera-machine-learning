function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

    % We can use matrix-vector multiplication to computer our sums
    sum1 = -log(h(theta, X))'*y - log(1-h(theta, X))'*(1-y) ;
    
    % Special case: don't regularize theta(1), therefore only use theta(2:end)
    reg1 = theta(2:end)'*theta(2:end);
    J = 1/m * sum1 + 0.5 * lambda/m * reg1;

    % Again, the sums for the gradient is computed using a matrix-vector mutliplication
    sum2 = X' * (h(theta, X) - y);
    reg2 = (lambda/m) * theta;
    % Special case: don't regularize theta(1)
    reg2(1) = 0.0;
    grad = 1/m * sum2 + reg2;


% =============================================================

end
