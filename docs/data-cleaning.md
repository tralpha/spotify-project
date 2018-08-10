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
Loading the data
The MPD is organized into 100 separate JSON files where each file contain 1'000 playlists. I order to give is flexibility we first load the data into four distict data structures:
1. A list containing all the playlist
2. A dictionary with all the tracks
3. A list that maps songs to playlists
4. A list that maps duplicate songs to playlists

The fourt data structure was added after we conducted the EDA. During the EDA we found out that a playlist can contain duplicate songs. Although this may make sense for someone that is creating a playlist we feel that it does not make sense for a suggestion engine. Suggesting something that has already been played feels like cheating. We put all the duplicate into the last data structure, that way we could see how many duplicated there were.

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

Vectorization, transfer to sparse matrix, merging of playlist and song data, creating negative samples to train on, creating massive track list to predict on.

We started by creating 3 separate data frames from the MPD: one for global playlist features (one row per unique playlist), one for track features (one row per unique track) and one to map tracks to playlists.  In order to explore the data on the fly, we limit the number of playlists loaded into our notebook to a few thousand.  Finally, we merge the data frames to create one frame, ‘merged’, that has one track per row. The features of each track are the features of the playlist from which it came from together with the features of the track itself. 

As an example, if we loaded n playlists each containing m songs, then our merged dataframe will have n x m rows.
After feature engineering, we create our label variable, which is called `match`. `match` is described below:
Match: All track rows are set ‘match’ = 1, signifying that they belong to their respective playlists.  In order to aid the training of any model, we also appended to ‘merged’ a negative sample dataframe. The negative sample frame is a copy of ‘merged’, but with all the tracks randomly assigned to playlists with ‘match’ column set to 0. Each playlist’s positive sample to negative sample ratio is 1:1.
