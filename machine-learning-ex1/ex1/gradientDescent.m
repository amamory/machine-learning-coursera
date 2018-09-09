function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
features = size(X,2);
theta_change = zeros(features,1);

%m = 3;
%X = [ones(m, 1), [1;2;3]]
%y = [1;2;3]
%theta = [0,1]
for iter = 1:num_iters
    h = X*theta;
    % ============================================================
    %1st method dependent of # of features
    %theta_change(1) = alpha * (1/m) * sum ((h-y)' * X(:,1));
    %theta_change(2) = alpha * (1/m) * sum ((h-y)' * X(:,2));
    %theta_change(3) = alpha * (1/m) * sum ((h-y)' * X(:,3));
    %theta = theta - theta_change;
    % ============================================================
    %2nd method independent of # of features
    theta_change = alpha * (1/m) * ((h-y)' * X);
    theta = theta - theta_change';
    % ===========================================================
    
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
