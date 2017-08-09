Movie recommendation filter--collaborative, item-based, and hybrid

This project uses the MovieLens Small dataset (https://grouplens.org/datasets/movielens/)

movieRec achieves matrix factorization by starting with random values and using a gradient descent algorithm.

movieRec_sklearn uses the singular value decomposition, utilizing scikit-learn's TruncatedSVD. It also includes a cross-validation analysis to see whether the model is significantly improved by adding features. 

The cross-validation analysis works by shuffling the data and splitting it into two halves, one for training the models and one for testing them. Each model is fit to the training data, resulting in a transformation of the movie ratings from the 671-dimensional user space into a three- or four-dimensional space (depending on the model). 
 To see how the models perform on the test data, I projected it into that same lower-dimensional space using the models fit to the training data. 
Then those projections are transformed BACK into the high-dimensional user-space, and these values are compared to the actual user ratings to determine the cost (I used the sum of squared differences) associated with each model.
 I interpret the difference in these cost values from one model to another as a metric of the improvement in performance. To determine whether the difference is statistically significant, I repeated the procedure 1,000 times on different random splits of the data to get a distribution. 
If 95% of the distribution is less than zero, it means the second model is better to a statistically significant degree.

Comparing the models this way automatically penalizes models with more parameters, because these will be more likely to over-fit the training data set, resulting in a worse fit to the test data.

As-is, this analysis evaluates whether the difference in the models' ability to predict user ratings is statistically significant, but not whether that difference is large enough to make a meaningful improvement in its actual use. 
(In other words, adding features might allow the model to consistently predict my ratings within 0.1 stars, which would be statistically significant, but might not affect which movies were actually recommended to me.)
 Given the context, it might make sense to change the cost function to reflect the model's actual use--e.g., instead of measuring the error in predicting users' exact ratings, just predict whether users will "like" each film (based on a reasonable definition of "like," e.g., one or two stars above the user's mean rating). 
 Optimizing over this cost function would prioritize the real goal of classifying each film as liked or not liked by each user.

content_filter uses a Bayesian analysis to predict users' movie ratings based on the genres of movies they've already rated. (A lot more information could be added based on other datasets, but I just used genre for the first pass.)

hybridSystem:
(1) uses the Bayesian analysis to predict all users' ratings for all unrated movies. 
(2) Then the dense matrix containing these predictions and the explicit ratings is used in the matrix factorization (SVD).

NEXT STEPS:
(1) Evaluate the hybrid system--does it actually do better than the others? (cross-validation to compare models)
(2) Clean up--scoop up all the code that's used repeatedly and define some methods that can be used on other datasets or different splits of the data...
(3) some way of optimizing the second step to account for the level of confidence in the predicted ratings? (e.g., implement a cost function that down-weights the model's error for some cases) (that means using something other than SVD to factor the matrix)

