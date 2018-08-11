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
[Conclusion and What's Next](https://tralpha.github.io/spotify-project/conclusion.html) <br>
[Literature Review](https://tralpha.github.io/spotify-project/literature-review.html) <br>

# Data Cleaning:
## Loading the data

The MPD is organized into 100 separate JSON files where each file contain 1'000 playlists. In order to give us flexibility we first load the data into four distict data structures:

1. A list containing all the *playlist*

2. A dictionary with all the *tracks*

3. A list that maps *songs* to *playlists*

4. A list that maps *duplicate songs* to *playlists*

The fourth data structure was added after we conducted the EDA. During the EDA we found out that a playlist can contain duplicate songs. Although this may make sense for someone that is creating a playlist manually, we feel that it does not make sense for a suggestion engine. Suggesting something that has already been played feels like cheating. We put all the duplicates into the last data structure, that way we could see how many duplicates there were and easily exclude them from further processing.

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

## Pandas

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
<img src="images/playlist_map_df.png" width="440">

## Negative Samples
**Here we first refer to our response variable "Match" which will be frequently referenced in our reporting**
<br>
In order to train our model we also need a negative sample set. We will add random songs to each playlist. We add a binary value called `match` which is used as the response variable for the model.<br>

We make a copy of the playlist to song mapping dataframe, we then randomly assign songs to playlist:
```python
playlist_map_df_negative = playlist_map_df.copy()
random = playlist_map_df.sample(n=len(playlist_map_df)).reset_index()
playlist_map_df_negative['track_uri'] = random['track_uri']
```

We verify that the new dataset is indeed scrambled by checking heads:
```python
playlist_map_df.head()
playlist_map_df_negative.head()
```
<img src="images/playlist_map_df_scrambled.png" width="440">

## Denormalizing the data
<br>
We cannot effectively use multiple data frames for our modelling. we next need to merge the data frames together. We first merge the dataset with the original data into a single data frame:
```python
merged = pd.merge(
    pd.merge(
        tracks_df, playlist_map_df, left_index=True, right_on='track_uri'),
    playlist_df,
    on='playlist_pid')
```

We also create a single dataframe of the scrambled *(negative samples)* data frame:
```python
negative_samples = pd.merge(
    pd.merge(
        tracks_df, playlist_map_df_negative, left_index=True, right_on='track_uri'),
    playlist_df,
    on='playlist_pid')
```
We now have two dataframes both with the same playlist, but the negative samples playlist has randomized its song contents. Before merging the datasets together we add the binary response variable `match`.  Each playlist will have a matching to not matching ratio of 1:1.
```python
negative_samples['match'] = 0
merged['match'] = 1
```

We merge the two datasets together:
```python
dataset = pd.concat([negative_samples, merged]).sort_values(by=['playlist_pid']).reset_index(drop=True)
```

## Using playlist name and description as a proxy for song meta-data
As described in the [Data Description](https://tralpha.github.io/spotify-project/data-description.html) we have chosen not to include the Million Song Dataset (MSD) as we believe we can extract meta-data on songs from the playlist name and the description. We show the the top 30 playlist names, all those playlists are single word playlists name and their names are indeed categorical like: rock, chill, country etc.

In order to ease the word vectorization we clean up the playlist names and the descriptions by removing stop words and we set everything to lowercase. We also remove the types of words we see often in playlist names like: "playlist", "favorites", "tunes", "good, "mix" etc. We do not think these words add any value for categorizing the songs.

```python
import nltk
import string
from nltk.corpus import stopwords
stop = stopwords.words('english')
ignored_words = [
    'music', 'songs', 'playlist', 'good', 'jams', 'mix', 'lit', 'best',
    'stuff', 'quot', 'like', 'one', 'amp', 'get', 'make', 'new', 'know',
    'really', 'back', 'day', 'days', 'little', 'things', 'great', 'everything',
    'jamz', 'tunes', 'artist', 'song', 'top', 'listen', 'favorite', 'bops',
    'description', 'top', 'ever', 'mostly', 'enjoy', 'bunch', 'track',
    'tracks', 'collection', 'need', 'every', 'favorites', 'may', 'got',
    'right', 'let', 'better', 'made'
]

def word_cleanup(df_col):
    df_col = df_col.apply(lambda x: x.lower())
    df_col = df_col.str.replace('[^a-z]+', ' ')
    df_col = df_col.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    df_col = df_col.apply(lambda x: ' '.join([word for word in x.split() if word not in (ignored_words)]))
    df_col = df_col.str.replace(r'\b\w{1,2}\b', '').str.replace(r'\s+', ' ')
    return df_col

playlist_df.playlist_description = word_cleanup(playlist_df.playlist_description)
playlist_df.playlist_name = word_cleanup(playlist_df.playlist_name)
```
