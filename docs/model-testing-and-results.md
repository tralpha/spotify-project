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
[Literature Review](https://tralpha.github.io/spotify-project/literature-review.html) <br>

Using the trained model, we proceed to recommend songs to our trained playlists. As mentioned, a `train_test_split` was done on our merged dataset to obtain an `X_train` and an `X_test`. This `train_test_split` was done in a `stratified` manner on the `playlist_id` feature, meaning that for every playlist in the data, 10% of songs were reserved for testing. This `train_test_split` strategy enabled us to simulate real world environments, where the Spotify team will have a seed playlist to start with, from which they'll recommend songs.

To make things run faster, we chose to predict whether a "test set track" belonged to a particular playlist. We did not do the prediction of playlist membership on the training set tracks, because our algorithm had already seen them so it will be a trivial task to predict on them. In addition, in order to reduce computation time and quickly get feedback on how our model was doing in a track recommendation environment, we chose to predict only on "test set tracks".

That being said, when 2000 playlists were loaded, there were still about 26000 unique test set tracks. Our test pipeline therefore consisted of appending all of these 26000 unique test set tracks to each and every playlist trained on, and predicting track membership. So, for the 2000 playlists, our model will need to do about 52 million predictions (26000 * 2000). 

For every playlist, we predicted track membershp for the 26000 test set tracks. This indeed takes a while to run, and one potential improvement will be to make this testing process to run faster, possibly by vectorizing the pipeline rather than using a `for` loop. 

We used our trained model to predict track membership for each playlist. After prediction, we ranked the tracks by probabilities, and then we calculated the `r_precision` with the **ground truth** tracks which we reserved for training. On average, playlists have about 50 tracks, so 5 tracks will be reserved for the testing pipeline to calculate this `r_precision` metric. The task indeed is not an easy one: out of 26000 tracks (using 2000 playlists), can the algorithm rank the tracks by probability, such that the ground_truth tracks present in the test set (which the algorithm didn't see before) are ranked first? This is the question r_precision answers. Out of the first x songs (where x is the number of ground truth predictions), how many ground truth songs were present? 

When using our model, our mean `r_precision` is about `0.12`. This means that out of 25 ground truth songs, 3 will be present in the top 25 predictions ranked by probability. We did some sanity checking to see how our model is doing compared to other Spotify RecSys Challenge participants, and we saw that the best model had an `r_precision` of `0.224656`. However we got `0.12` just based on 2000 playlists, and they got `0.224656` based on training on the whole MPD Dataset.

After obtention of the `r_precision` metric for our system, we proceeded to test our model on a sample playlist built by John, our teammate. As mentioned before, this required us to re-run the whole training and testing pipeline (this is a major limitation of our model). The recommendations provided were quite reasonable, according to John.

