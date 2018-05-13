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
a=1;
cost = 0;
t=0;
regu = 0;
for j = 2:size(theta,1)
	regu = regu + theta(j)*theta(j);
end
regu=(regu*lambda)/(2*m);

% fprintf('%f',regu);
for i = 1:m
		one= y(i)* log( sigmoid(transpose(theta)*transpose(X(i,:)) ));
		cost = cost - ( one + (1-y(i))*log(1-sigmoid(transpose(theta)*transpose(X(i,:))) ));
end
% fprintf('\n%f',cost);

J=(cost/m) + regu;
% grad(1)=grad(1) + (a)*((1/m)*transpose(X(:,1))*( sigmoid( X(:,1)*theta(1,:) ) - y(1) ));
qw=(transpose((X)) * (sigmoid(X*theta) - y));
% fprintf('\n%f',grad(2)-(a*(1/m)*qw(2)));

% fprintf('\n%f',transpose((X)) * (sigmoid(X*theta) - y));
grad(1)=grad(1) + (a)*((1/m)*(transpose(X)*( sigmoid(X*theta) - y ))(1));
% fprintf('\n%f',grad(2));
% fprintf('\n%f',size(transpose(X(:,2:size(X,2)))));
grad(2:size(theta,1),:)=grad(2:size(theta,1),:) + ((a) * ((1/m)*(transpose(X)*(sigmoid(X*theta)-y))(2:size(X,2),:) + (lambda/m)* theta(2:size(theta,1),:)));

% =============================================================

end