function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%

%size(X) 
%size(y) 
%size(all_theta)
%num_labels 
%lambda

initial_theta = zeros(n+1,1);
options = optimset('GradObj', 'on', 'MaxIter', 50);

%%COST WITH 50 ITERATIONS
%Training One-vs-All Logistic Regression...
%Iteration    50 | Cost: 1.375660e-02
%Iteration    50 | Cost: 5.725248e-02
%Iteration    50 | Cost: 6.400192e-02
%Iteration    50 | Cost: 3.643481e-02
%Iteration    50 | Cost: 6.186508e-02
%Iteration    50 | Cost: 2.171851e-02
%Iteration    50 | Cost: 3.536424e-02
%Iteration    50 | Cost: 8.584111e-02
%Iteration    50 | Cost: 7.883111e-02
%Iteration    50 | Cost: 1.001876e-02
%Program paused. Press enter to continue.
%
%Training Set Accuracy: 94.880000
%>>
%
%COST WITH 100 ITERATIONS
%Training One-vs-All Logistic Regression...
%Iteration   100 | Cost: 1.316702e-02
%Iteration   100 | Cost: 5.347617e-02
%Iteration   100 | Cost: 5.990606e-02
%Iteration   100 | Cost: 3.347102e-02
%Iteration   100 | Cost: 5.699809e-02
%Iteration   100 | Cost: 1.880249e-02
%Iteration   100 | Cost: 3.112734e-02
%Iteration   100 | Cost: 8.092878e-02
%Iteration   100 | Cost: 7.401799e-02
%Iteration   100 | Cost: 8.583884e-03
%Program paused. Press enter to continue.
%
%Training Set Accuracy: 95.800000
%>>

for label = 1:num_labels
	# get the y`s related to each label
	[all_theta(label,:)] = fmincg(@(t)(lrCostFunction(t, X, y==label, lambda)), initial_theta, options);
end

% =========================================================================


end
