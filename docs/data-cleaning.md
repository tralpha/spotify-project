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

# Data Cleaning:
**Loading the data**
<br>
The MPD is organized into 100 separate JSON files where each file contain 1'000 playlists. In order to give us flexibility we first load the data into four distict data structures:
1. A list containing all the <b>playlist</b>
2. A dictionary with all the <b>tracks</b>
3. A list that maps <b>songs to playlists</b>
4. A list that maps <b>duplicate songs to playlists</b>

The fourt data structure was added after we conducted the EDA. During the EDA we found out that a playlist can contain duplicate songs. Although this may make sense for someone that is creating a playlist manually, we feel that it does not make sense for a suggestion engine. Suggesting something that has already been played feels like cheating. We put all the duplicates into the last data structure, that way we could see how many duplicates there were and easily exclude them from further processing.

```python
playlists = list()
tracks = dict()
map_pl = list()
map_pl_duplicate = list()
max_files_for_quick_processing = 4

def process_track(track):
    key = track['track_uri']
    if not key in tracks:
        tk = dict()
        tk['track_artist_name'] = track['artist_name']
        tk['track_artist_uri'] = track['artist_uri']
        tk['track_name'] = track['track_name']
        tk['track_album_uri'] = track['album_uri']
        tk['track_duration_ms'] = track['duration_ms']
        tk['track_album_name'] = track['album_name']
        tk['track_pos'] = track['pos']
        tracks[track['track_uri']] = tk
    return key

def process_playlist(playlist):
    pl = dict()
    pl['playlist_name'] = playlist['name']
    pl['playlist_collaborative'] = playlist['collaborative']
    pl['playlist_pid'] = playlist['pid']
    pl['playlist_modified_at'] = playlist['modified_at']
    pl['playlist_num_albums'] = playlist['num_albums']
    pl['playlist_num_tracks'] = playlist['num_tracks']
    pl['playlist_num_followers'] = playlist['num_followers']
    pl['playlist_num_edits'] = playlist['num_edits']
    pl['playlist_duration_ms'] = playlist['duration_ms']
    pl['playlist_num_artists'] = playlist['num_artists']
    if 'description' in playlist:
        pl['playlist_description'] = playlist['description']
    else:
        pl['playlist_description'] = ''
    trks = set()
    for track in playlist['tracks']:
        if track['track_uri'] not in trks:
            trks.add(track['track_uri'])
            process_track(track)
            map_pl.append([playlist['pid'], track['track_uri']])
        else:
            map_pl_duplicate.append([playlist['pid'], track['track_uri']])
    return pl

def process_mpd(path):
    count = 0
    filenames = os.listdir(path)
    for filename in sorted(filenames):
        print(filename)
        if filename.startswith("mpd.slice.") and filename.endswith(".json"):
            fullpath = os.sep.join((path, filename))
            f = open(fullpath)
            js = f.read()
            f.close()
            slice = json.loads(js)
            for playlist in slice['playlists']:
                playlists.append(process_playlist(playlist))
            count += 1
            if quick and count > max_files_for_quick_processing:
                break

quick = True
process_mpd('mpd.v1/data')
```
After loading up the entire dataset we have:<br>
<b>1'000'000</b> playlists<br>
<b>2'262'292</b> tracks<br>
<b>65'464'776</b> songs in the playlists<br>
<b>881'652</b> duplicate songs<br>

**Pandas**
<br>
We now convert our three data structures to Pandas data frames:
```python
playlist_df = pd.DataFrame(playlists)
tracks_df = pd.DataFrame.from_dict(tracks, orient='index')
playlist_map_df = pd.DataFrame(map_pl, columns=['playlist_pid', 'track_uri'])
```

<b>The playlist dataframe</b>
```python
playlist_df.head()
playlist_df.tail()
```
![fig1](images/playlist_df.png)

<b>The tracks dataframe</b>
```python
tracks_df.head()
tracks_df.tail()
```
![fig2](images/tracks_df.png)

<b>The playlist to song mapping dataframe</b>
```python
playlist_map_df.head()
playlist_map_df.tail()
```
<img src="images/playlist_map_df.png" width="400">

**Negative Samples**
<br>
In order to train our model we also need a negative sample set. We will add random songs to each playlists. We add a binary value called "Match" which is used as the response variable for the model.<br>

We make a copy of the playlist to song mapping dataframe, we then randomly assign songs to playlist:
```python
playlist_map_df_negative = playlist_map_df.copy()
random = playlist_map_df.sample(n=len(playlist_map_df)).reset_index()
playlist_map_df_negative['track_uri'] = random['track_uri']
```

We verify that the new dataset is indeed scrambled
```python
playlist_map_df.head()
playlist_map_df_negative.head()
```
![fig4](images/playlist_map_df_scrambled.png)

**Denormalizing the data**
<br>
We cannot use multiple data frames for our midelling, we next need to merge the data frame together. We first merge the dataset with the original data into a single data frame
```python
merged = pd.merge(
    pd.merge(
        tracks_df, playlist_map_df, left_index=True, right_on='track_uri'),
    playlist_df,
    on='playlist_pid')
```

We also create a single dataframe of the scrambled (negative samples) data frame:
```python
negative_samples = pd.merge(
    pd.merge(
        tracks_df, playlist_map_df_negative, left_index=True, right_on='track_uri'),
    playlist_df,
    on='playlist_pid')
```
We now have two dataframe both with the same playlist, but the negative samples playlist has randomized its song contents. Before merging the datasets together we add the binary response variable:
```python
negative_samples['match'] = 0
merged['match'] = 1
```

We merge the two datasets together
```python
dataset = pd.concat([negative_samples, merged]).sort_values(by=['playlist_pid']).reset_index(drop=True)
```


```python
data_x = dataset.loc[:, dataset.columns != 'match']
data_y = dataset.match
data_train, data_test, y_train, y_test = train_test_split(
    data_x,
    data_y,
    test_size=0.3,
    stratify=dataset[['playlist_pid', 'match']],
    random_state=42,
    shuffle=True)
```

Vectorization, transfer to sparse matrix, merging of playlist and song data, creating negative samples to train on, creating massive track list to predict on.

We started by creating 3 separate data frames from the MPD: one for global playlist features (one row per unique playlist), one for track features (one row per unique track) and one to map tracks to playlists.  In order to explore the data on the fly, we limit the number of playlists loaded into our notebook to a few thousand.  Finally, we merge the data frames to create one frame, ‘merged’, that has one track per row. The features of each track are the features of the playlist from which it came from together with the features of the track itself. 

As an example, if we loaded n playlists each containing m songs, then our merged dataframe will have n x m rows.
After feature engineering, we create our label variable, which is called `match`. `match` is described below:
Match: All track rows are set ‘match’ = 1, signifying that they belong to their respective playlists.  In order to aid the training of any model, we also appended to ‘merged’ a negative sample dataframe. The negative sample frame is a copy of ‘merged’, but with all the tracks randomly assigned to playlists with ‘match’ column set to 0. Each playlist’s positive sample to negative sample ratio is 1:1.
