% Define a custom loss function that combines labeled and pseudo-labeled data
function loss = customLoss(y_true, y_pred, y_pseudo, lambda)
    labeledLoss = crossentropy(y_true, y_pred);
    pseudoLoss = crossentropy(y_pseudo, y_pred);
    
    loss = lambda * labeledLoss + (1 - lambda) * pseudoLoss;
end