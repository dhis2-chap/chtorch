# Adin meeting

## Questions

### Convergence
- Should eta be as standard normal as possible
- Are there tricks to make negative binomial more well behaved
- Feature compression architecture
- Using validation sets for time series data
- - Learning rate schedueling, early stopping, hp optimization etc.
- How important is the simplicity of the transform functions.
- How to make ar features for count data

### Framework



## Gridded predictors, accumulated targets
- Predictors come in a regular grid
- Targets come accumulated by regions
- Make a network to calculate eta for each grid cell
- Accumulate etas (using trainable pooling function) to an eta per region
- Predict counts basd on NB(theta) where \theta = f(eta)


## Notes
# Gating/mixture of experts/ routing
# Focal losses
