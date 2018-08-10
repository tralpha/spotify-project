# Table of Contents
[Project Statement and Goals](https://john-daciuk.github.io/spotify/project-statement-and-goals.html) <br>
[Motivation and Background](https://john-daciuk.github.io/spotify/motivation-and-background.html) <br>
[Data Description](https://john-daciuk.github.io/spotify/data-description.html) <br>
[EDA](https://john-daciuk.github.io/spotify/eda.html) <br>
[Data Cleaning](https://john-daciuk.github.io/spotify/data-cleaning.html) <br>
[Metrics](https://john-daciuk.github.io/spotify/metrics.html) <br>
[Data Cleaning](https://john-daciuk.github.io/spotify/data-cleaning.html) <br>
[Model Training](https://john-daciuk.github.io/spotify/model-training.html) <br>
[Interpreting the Model](https://john-daciuk.github.io/spotify/interpreting-the-model.html) <br>
[Model Testing and Results](https://john-daciuk.github.io/spotify/model-testing-and-results.html) <br>
[Literature Review](https://john-daciuk.github.io/spotify/literature-review.html) <br>

# Metrics:

Metrics for this task can be daunting:  with over 2 million unique songs in the MPD, how could any model manage to pick the one you yourself would have chosen to add to your playlist?  On the other hand, we could imagine a model that just picks all 2 million songs to add to your playlist and surely all of your favorite songs will be waiting for you.  To find a compromise, we created a training set with 90% of the tracks in each playlist and a test set with 10%.  For each playlist our model will choose a number of top song predictions equal in size to the test set for that playlist data.  The fraction of our top predictions that are actually in the test set is our metric.

This all sounds all the more reasonable in light of the fact that this is one of the metrics that the Spotify MPD Competition is actually using!  The name of the metric is R-Precision and the idea is widely used in testing recommendation systems.  

R-Precision Metric = *(Intersection of Prediction and Truth) / (Truth)* , where *(Truth)* is the set of tracks matching tracks in the test set portion of a playlist, and *Prediction* is the set of top predicted tracks, constrained to be equal in length to *Truth*.

We imported the Spotify Competition metrics library into our notebook.
