
function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

    % We can use matrix-vector multiplication to computer our sums
    sum1 = -log(h(theta, X))'*y - log(1-h(theta, X))'*(1-y);
    J = 1/m * sum1;
    % Again, the sum is computed using a matrix-vector mutliplication
    sum2 = X' * (h(theta, X) - y);
    grad = 1/m * sum2;  






% =============================================================

end



