% Define a custom output function to display gradients and other information
function customOutputFunction(info, net, imdsUnlabeled, lambda)
    if isfield(info, 'GradientDecay')
        fprintf('Gradient Decay: %.4f\n', info.GradientDecay);
    end
    
    % Use the current model to predict labels for unlabeled data
    predictedLabels = classify(net, imdsUnlabeled);
    
    % Convert categorical labels to numeric labels
    predictedLabelsNumeric = double(predictedLabels);
    
    % Calculate confidence scores (e.g., max softmax probability)
    confidenceScores = max(softmax(predictedLabelsNumeric), [], 2);
    
    % Filter confident predictions based on the confidence threshold
    confidenceThreshold = 0.9; % Adjust the confidence threshold as needed
    confidentIndices = confidenceScores > confidenceThreshold;
    
    % Display the number of confident pseudo-labels
    fprintf('Number of Confident Pseudo-Labels: %d\n', sum(confidentIndices));
    
    % Update the loss function with the confident pseudo-labels
    options.CustomLoss = @(y_true, y_pred)customLoss(y_true, y_pred, predictedLabels(confidentIndices), lambda);
end
