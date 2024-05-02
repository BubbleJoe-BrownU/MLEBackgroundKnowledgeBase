# Machine Learning

## Shapley Value

A prediction can be explained by assuming that each feature value of the instance is a "player" in a game where the prediction is the payout. Shapley values - a method from coalitional game theory - tells us how to fairly distribute the "payout" among the features.

The Shapley value is the average contribution of a feature value to the prediction in different coalitions. The Shapley value is NOT the difference in prediction when we would remove the feature from the model.

Here is what a linear model prediction looks like for one data instance:
$$
\hat f(x) = \beta_0 + \beta_1x_1 + ... + \beta_px_p
$$
The contribution $\phi_j$ of the j-th feature on the prediction $\hat f(x)$ is:
$$
\phi_j(\hat f) = \beta_jx_j - E(\beta_jX_j) = \beta_jx_j - \beta_jE(X_j)
$$
Where $E(\beta_jX_j)$​ is the mean effect estimate for feature j. The contribution is the difference between the feature effect minus the average effect. Now we know how much each feature contributed to the prediction.

If we sum all the feature contributions for one instance, the result is the following:



#### Beeswarm plot

The beeswarm plot is designed to display an information-dense summary of how the top features in a dataset impact the model's output. Each instance the given explanation is represented by a single dot on each feature row. The x position of the dot is determined by the SHAP value of that feature, and dots pile up along each feature row to show density. Color is used to display the **original value of a feature**. Therefore, we can interpret how the change of feature value might possibly affect the model prediction.

By default the features are ordered using `shap_values.abs.mean(0)`, which is the mean absolute value of the SHAP values for each feature



## Bias/Variance Trade-off

Bias is the **average magnitude of the model’s error**, averaged over different testing/training data sets. It could refer to the magnitude difference between the expected value and the true value of the thing being estimated, or it could just be 1 if incorrect and 0 if correct. The bias error is an error from **erroneous assumptions** in the learning algorithm

Variance is a measure of the **spread of the magnitude of the error**, averaged over different testing/training data sets. The variance is an error from **sensitivity to small fluctuations** in the data.

When a model is underfitting, it typically has a high bias and low variance because it does not have enough complexity to capture important structures in the training data. When a model is overfitting, it typically has a low bias and high variance as it's too comlex and fit unimportant structures or noises in the training data. The trade-off is really about pick a reasonable complexity for a model such that it fits well on the training data and **generalizes** well to unseen data.

## How to handle categorical variables

### Ordinal features

### Categorical features



## How to measure the performance of a classification model?

### Recall and Precision

![Recall and Precision](https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Precisionrecall.svg/800px-Precisionrecall.svg.png)
$$
\text{precision} = \frac {\text{TP}} {\text{TP} + \text{FP}} \\
\text{recall} = \frac {\text{TP}} {\text{TP} + \text{FN}}
$$

### AUC-ROC
AUC stands for "Area under the ROC Curve"

### Accuracy

```python
def accuracy_score(y_true, y_pred, normalize=True, sample_weight=None):
  """Accuracy classification score
  In multilabel classification, this function computes subset accuracy:
   the set of labels predicted for a sample must *exactly* match the corresponding set of labels in y_true
  """
  
```



### Balanced Accuracy

The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets. It is defined as **the average of recall obtained on each class**.
$$
\text{balanced accuracy} = 
$$

```python
def balanced_accuracy_score(y_true, y_pred, sample_weight=None, adjusted=False):
  """
  adjusted: when true, the result is adjusted for chance, so that random performance would score 0, while keeping perfect performance at a score of 1.
  """
  C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
  with np.errstate(divide="ignore", invalid="ignore"):
    per_class = np.diag(C) / C.sum(axis=1)
    
```




## Overfitting

- How to manage over-fitting

data augmentation (like RandAug)

regularization (l1 regularization, l2 regularization, dropout, cutout, stochastic depth)

pick a better hypothesis space (change the model from CNNs to Transformers)

## Active Learning



## Federate Learning



## Recommendation System

