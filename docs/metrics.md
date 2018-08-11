### Table of Contents
[Project Statement and Goals](https://tralpha.github.io/spotify-project/project-statement-and-goals.html) <br>
[Motivation and Background](https://tralpha.github.io/spotify-project/motivation-and-background.html) <br>
[Data Description](https://tralpha.github.io/spotify-project/data-description.html) <br>
[EDA](https://tralpha.github.io/spotify-project/eda.html) <br>
[Data Cleaning](https://tralpha.github.io/spotify-project/data-cleaning.html) <br>
[Metrics](https://tralpha.github.io/spotify-project/metrics.html) <br>
[Model Training](https://tralpha.github.io/spotify-project/model-training.html) <br>
[Interpreting the Model](https://tralpha.github.io/spotify-project/interpreting-the-model.html) <br>
[Model Testing and Results](https://tralpha.github.io/spotify-project/model-testing-and-results.html) <br>
[Conclusion and What's Next](https://tralpha.github.io/spotify-project/conclusion.html) <br>
[Literature Review](https://tralpha.github.io/spotify-project/literature-review.html) <br>

# Metrics:

## Introduction

Metrics for this task can be daunting:  with over 2 million unique songs in the MPD, how could any model manage to pick the one you yourself would have chosen to add to your playlist?  On the other hand, we could imagine a model that just picks all 2 million songs to add to your playlist and surely all of your favorite songs will be waiting for you.  To find a compromise, we created a training set with 70% of the tracks in each playlist and a test set with 30%.  For each playlist, our model will choose a number of top song predictions.  These predictions will then be narrowed down to the ones with greatest probability of match and be equal in size to the test set for that playlist data.  The fraction of our model's top predictions that are actually in the test set is one of our metrics.

This all sounds all the more reasonable in light of the fact that this is one of the metrics that the Spotify MPD Competition is actually using!  The name of the metric is **R-Precision** and the idea is widely used in testing recommendation systems.  We imported the Spotify Competition metrics library into our notebook.  

## R-Precision Mathematically Defined
R-Precision Metric = *(Intersection of Prediction and Truth) / (Truth)* , where *(Truth)* is the set of tracks matching tracks in the test set portion of a playlist, and *Prediction* is the set of top predicted tracks, constrained to be equal in length to *Truth*.

R-Precision is a reasonable metric because it has a simple formula that makes for easy interpretability and it also allows us to compare to the scores acheived in the Spotify MPD Competition.  However, we also wanted another metric that was engaging.

## Our Second Metric:  See Top 10 Recommendations for a Given Playlist
Of course we also added functionality in the testing phase so that we could inspect predictions being made for any given `playlist_pid` and see for ourselves if the recommendations are complimentary.  We also have the following function which allows us to create our own playlist and append it to the main dataframe:

```python
def create_my_playlist(songs: list, name: string, description: string):
    my_playlist = pd.DataFrame()
    for song in songs:
        # Append to new playlist the song as it first appears in the main dataframe merged along with its meta information
        my_playlist = my_playlist.append(merged.loc[merged.track_name == song].iloc[0]           [tracks_df.columns.append(pd.Index(["track_uri"]))])
    
    # Fill in playlist meta info for all newly added songs
    playlist_info = pd.DataFrame(columns = list(playlist_df.columns))
    playlist_info["playlist_collaborative"] = pd.Series("false")
    playlist_info["playlist_description"] = description
    playlist_info["playlist_modified_at"] = 1496793600
    playlist_info["playlist_name"] = name
    playlist_info["playlist_num_edits"] = 1
    playlist_info["playlist_num_followers"] = 1
    playlist_info["playlist_pid"] = np.max(merged.playlist_pid) + 1
    playlist_info["playlist_num_artists"] = my_playlist["track_artist_uri"].nunique()
    playlist_info["playlist_num_tracks"] = len(my_playlist)
    playlist_info["playlist_duration_ms"] = np.sum(my_playlist["track_duration_ms"])
    playlist_info["playlist_num_albums"] = my_playlist["track_album_uri"].nunique()
    playlist_info = pd.concat([playlist_info] * len(songs))
    playlist_info.index = list(my_playlist.index)
    
    # Concat new song playlist track df with playlist specific columns
    result_df =  pd.concat([my_playlist, playlist_info], axis=1)
    result_df.index = range(len(merged), len(merged) + len(songs))
    
    #Return new playlist dataframe which can now be appended to the main MPD dataframe we are working with
    return result_df
```
