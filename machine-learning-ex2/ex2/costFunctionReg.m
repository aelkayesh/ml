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
temp = theta;
temp(1) = 0;
seg = sigmoid(X*theta);
T1 = (-1.*y) .* log(seg);
T2 = (1 .- y) .* log( 1 - seg);
J = sum(T1 .- T2) / m;
reg_term = (lambda*(sum(power(temp, 2)))) / (2 * m);
J = J + reg_term; 

 
grad = X'*(seg - y);
grad = (grad ./m) ;
grad = grad .+(lambda /m) .*temp;











% =============================================================

end
