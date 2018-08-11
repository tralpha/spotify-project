## Table of Contents
[Project Statement and Goals](https://tralpha.github.io/spotify-project/project-statement-and-goals.html) <br>
[Motivation and Background](https://tralpha.github.io/spotify-project/motivation-and-background.html) <br>
[Data Description](https://tralpha.github.io/spotify-project/data-description.html) <br>
[EDA](https://tralpha.github.io/spotify-project/eda.html) <br>
[Data Cleaning](https://tralpha.github.io/spotify-project/data-cleaning.html) <br>
[Metrics](https://tralpha.github.io/spotify-project/metrics.html) <br>
[Model Training](https://tralpha.github.io/spotify-project/model-training.html) <br>
[Interpreting the Model](https://tralpha.github.io/spotify-project/interpreting-the-model.html) <br>
[Model Testing and Results](https://tralpha.github.io/spotify-project/model-testing-and-results.html) <br>
[Conclusion and Literature Review](https://tralpha.github.io/spotify-project/literature-review.html) <br>

# Model Testing and Results

## Introduction
Using the trained model, we proceed to recommend songs to our trained playlists. As mentioned, a `train_test_split` was done on our merged dataset to obtain an `X_train` and an `X_test`. This `train_test_split` was done in a `stratified` manner on the `playlist_id` feature, meaning that for every playlist in the data, 30% of songs were reserved for testing. This `train_test_split` strategy enabled us to simulate real world environments, where the Spotify team will have a seed playlist to start with, from which they'll recommend songs.  In the future we'd like to experiment with creating our own train-test splits rather than using the sklearn library to get more intuition on the process and a finer degree of control.  See **Model Training** for an inspection of the data split.

To make things run faster, we chose to predict *only* whether a "test set track" belonged to a particular playlist. We did not do the prediction of playlist membership on the training set tracks: our algorithm had already seen them so it will be a trivial task to predict on the tracks in the playlist in the training set. In addition, predicting on only test set tracks was also helpful in reducing computation time and quickly getting feedback on how our model performs in a track recommendation environment.  In the future we would like to predict on training tracks that were **not** in the playlist we are currently making recommendations for.

The above caveat beside, when 2000 playlists were loaded, there were still about 26000 unique test set tracks. Our test pipeline therefore consisted of appending all of these 26000 unique test set tracks to each and every playlist trained on and predicting track membership. For the 2000 playlists, our model will need to do about *52 million* predictions (26000 * 2000). 

For every playlist, we predict track membershp for the 26000 test set tracks. Runtime is slow; one potential improvement could be vectorizing the pipeline rather than using a `for` loop. 

## Test Code
```python
r_precisions = []
pbar = tqdm(data_test.groupby(['playlist_pid']))
for pid, df in pbar:
    p_info = df[playlist_df.columns].iloc[0]
    labels = y_test.loc[df.index]
    
    # Positive Tracks
    positive_tracks_idx = labels[labels == 1].index
    positive_tracks = data_test.loc[positive_tracks_idx]
    sp_positive_tracks = vectorizer.transform(positive_tracks.values)
    
    # Negative Tracks
    negative_tracks_idx = ~np.isin(data_test.index, positive_tracks_idx)
    negative_tracks = data_test[negative_tracks_idx].drop(
        playlist_df.columns, axis=1)
    negative_playlist = np.array([p_info.values] * len(negative_tracks))
    negative_playlist_samples = np.hstack([negative_tracks, negative_playlist])
    sp_negative_tracks = vectorizer.transform(negative_playlist_samples)
    # from IPython.core.debugger import set_trace; set_trace()
    
    # Test Tracks
    test_tracks = vstack([sp_negative_tracks, sp_positive_tracks])
    index_order = negative_tracks.index.append(positive_tracks_idx)
    
    # Predict, r_precision
    y_prob = clf.predict_proba(test_tracks)
    # from IPython.core.debugger import set_trace; set_trace()
    y_pred = np.argsort(-y_prob[:,1])
    best_pred = index_order[y_pred]
    if len(positive_tracks_idx) > 0:
        r_precisions.append(r_precision(positive_tracks_idx, best_pred))
    pbar.set_description("{}".format(np.mean(r_precisions)))
```

## Results
After prediction, we ranked the tracks by probabilities, and then we calculated the `r_precision` with the **ground truth** tracks which we reserved for training. On average, playlists have about 50 tracks, so 5 tracks will be reserved for the testing pipeline to calculate this `r_precision` metric. The task indeed is not an easy one: out of 26000 tracks (using 2000 playlists), can the algorithm rank the tracks by probability, such that the ground_truth tracks present in the test set (which the algorithm didn't see before) are ranked first? This is the question r_precision answers. Out of the first x songs (where x is the number of ground truth predictions), how many ground truth songs were present? 

When using our model, our mean `r_precision` is about `0.12`. This means that out of 25 ground truth songs, 3 will be present in the top 25 predictions ranked by probability. We did some sanity checking to see how our model is doing compared to other Spotify RecSys Challenge participants, and we saw that the best model had an `r_precision` of `0.224656`. However we got `0.12` just based on 2000 playlists, and they got `0.224656` based on training on the whole MPD Dataset.

After obtention of the `r_precision` metric for our system, we proceeded to test our model on a sample playlist built by John, our teammate. As mentioned before, this required us to re-run the whole training and testing pipeline (this is a major limitation of our model). The recommendations provided were quite reasonable, according to John.

