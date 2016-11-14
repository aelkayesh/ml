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

sumtheta = 0;

for k=2:size(theta),
  sumtheta += power(theta(k),2);
end
sumtheta .*= (lambda/ (2*m));
seg = sigmoid(X*theta);
T1 = (-1.*y) .* log(seg);
T2 = (1 .- y) .* log( 1 - seg);
J = (sum(T1 .- T2) / m) + sumtheta; 


for j = 1: size(theta),
  for i = 1: m,    
    grad(j) +=((sigmoid(theta'*X(i,:)') - y(i)) * X(i,j));
  end
  grad(j) /= m;
  if j > 1
    grad(j) += ((lambda/m) * theta(j));
  end
end






% =============================================================

end
