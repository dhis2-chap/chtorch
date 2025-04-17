# Project Ideas

## Capacity/regularization aware HPO

Register the parameter as either a capacity or regularization hyperparameter.
When sampling, assume that each point in hyperparameter space is either on the under- or overfitting stage.
Calculate the probability of a  point being on each side and sample a new point in HP space based on that.
This means that the alogrithm tries to avoid increasing the regularization if it seems to be underfitting etc.
Could possibly also classify the curve of validation loss as either overfitting or underfitting.

### Implementation
- Always look at the full validation curve, and use the last value as score
- Also look at the training curve
- Model capacity as one spline and gap as another



