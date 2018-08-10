## Table of Contents
[Project Statement and Goals](https://tralpha.github.io/spotify-project/project-statement-and-goals.html) <br>
[Motivation and Background](https://tralpha.github.io/spotify-project/motivation-and-background.html) <br>
[Data Description](https://tralpha.github.io/spotify-project/data-description.html) <br>
[EDA](https://tralpha.github.io/spotify-project/eda.html) <br>
[Data Cleaning](https://tralpha.github.io/spotify-project/data-cleaning.html) <br>
[Metrics](https://tralpha.github.io/spotify-project/metrics.html) <br>
[Data Cleaning](https://tralpha.github.io/spotify-project/data-cleaning.html) <br>
[Model Training](https://tralpha.github.io/spotify-project/model-training.html) <br>
[Interpreting the Model](https://tralpha.github.io/spotify-project/interpreting-the-model.html) <br>
[Model Testing and Results](https://tralpha.github.io/spotify-project/model-testing-and-results.html) <br>
[Literature Review](https://tralpha.github.io/spotify-project/literature-review.html) <br>

Once we have obtained our negative samples dataframe, we then merge them with our positive samples dataframe. This forms a big dataframe containing positive samples of our playlist+tracks features (with `match`=1) and negative samples of our playlist+track features (songs which are not supposed to be in a playlist, with `match`=1)

This merged dataframe is split into a training and test set using the `sklearn` `train_test_split` function. Because we have a large dataset, we choose to reserve 10% of our dataset for testing purposes. 

This train dataframe (called `data_train` in code) is then vectorized to obtain features to be trained using our model. We vectorize because our algorithm cannot be ran directly on text, so we need to extract features before training. We do the feature extraction using a `FeatureUnion` class from sklearn, which concatenates together binary features from different main text features, from the original merged dataset. The feature we consider for the problem are:

1. `track_artist_uri`
2. `track_album_uri`
3. `track_uri`
4. `playlist_pid`
5. `playlist_name`
6. `playlist_description`
7. `track_duration_ms`

The vectorizer is fit on the training set, to obtain a big matrix of features, `X_train`. This `X_train` matrix contains about 106000 features when 2000 playlists are loaded, together with 241000 observations. We believe as more playlists are loaded, the observations to features ratio increases, which should make it easier for the algorithm to distinguish between positive and negative playlist+track concatenations after training.

We then transform our `X_test` matrix to obtain a similar matrix, with the same number of features as the `X_train`. 

After both the training and test matrices are obtained, we build a grid search pipeline using `GridSearchCV` to obtain an optimized version of `AdaBoostClassifier`, the algorithm we chose to use for our model. We chose Adaboost because by default it's a non-linear model, and we strongly believe that due to the number of features and the number of observations in the dataset, the decision boundary will be non-linear. Adaboost is also an ensemble algorithm, so it tends to outperform other simpler algorithms, which is another reason we chose to use it. 

The `GridSearchCV` takes some time to run, but finally outputs our optimized classifier, which has parameters: `n_estimators: ` and `learning_rate: `.

This optimized model gives us training accuracy of `x.x`, and test set accuracy of `x.x`. What this means is that on the training set, the model learns how to perfectly distinguish between tracks which should belong to a playlist and tracks which should not belong to a playlist. However, on the test set, the model is still doing some errors. Looking at the model's performance on the test set, we thought this was good enough, and decided to proceed to using the algorithm to actually recommend songs, and obtain an `r_precision` score on our recommender model. How we do this exactly is explained in the next section, `model-testing-and-results.md`.
