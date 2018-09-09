function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
%J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

%J = (1/2*m) * 
%m = 5;
%X = X(1:5,1:2);
%y = y (1:5,1);
%m = 3;
%X = [ones(m, 1), [1;2;3]]
%y = [1;2;3]
%theta = [0,1]
%h = theta(1)* X(:,1) + theta(2) * X(:,2);
h = X * theta;
%h - y
%(h - y) .^ 2
%sum((h - y) .^ 2)
J = sum((h - y) .^ 2) / (2*m);


% =========================================================================

end
