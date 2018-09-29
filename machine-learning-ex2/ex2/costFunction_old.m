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
debug = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

%fprintf('\n\n\nAMORY\n\n\n');
h = sigmoid (X * theta);
%size(h)
%log(h)
h1 = (-y)' * log(h);
h2 = (1-y)' * log(1-h);
J = sum(h1-h2) / m;

grad = (1/m) * ((h-y)' * X)
return

%size(J);
%J
%fprintf('\n\n\nGRAD\n\n\n');

%function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
alpha = 0.003;
num_iters = 50;

if debug == 1
	J_history = zeros(num_iters, 1);
end
features = size(X,2);
theta_change = zeros(features,1);


for iter = 1:num_iters
    h = sigmoid (X*theta); 
    %2nd method independent of # of features
    theta_change = alpha * (1/m) * ((h-y)' * X);
    theta = theta - theta_change';
    % ============================================================
	if debug == 1
		% Save the cost J in every iteration    
		h1 = (-y)' * log(h);
		h2 = (1-y)' * log(1-h);
		%J = sum(h1-h2) / m;
		%J_history(iter) = computeCostMulti(X, y, theta);
		J_history(iter) = sum(h1-h2) / m;
	end

end

grad = theta;

if debug == 1
	
	h = sigmoid (X * theta);
	h1 = (-y)' * log(h);
	h2 = (1-y)' * log(1-h);
	cost_grad = sum(h1-h2) / m;
	fprintf('Theta found by gradient descent:\n');
	fprintf('%f\n', theta);
	fprintf('Cost of theta: %f\n', cost_grad);


	%Plot the cost reduction in the  gradientDescent
	hold on; % keep previous plot visible
	plot([1:num_iters], J_history, '-')
	legend('iterations', 'cost')
	hold off % don't overlay any more plots on this figur
	fprintf('\n\n\nEND GRAD\n\n\n');
end
% =============================================================

end
