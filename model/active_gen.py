from sklearn.svm import LinearSVC

import numpy as np
import requests


app_url = "http://app:8000"


def get_tracks_by_ids(track_ids):
    tracks = (requests.post(f"{app_url}/track_by_id", json=track_ids)).json()
    return tracks


def get_tracks_without_mentioned_by_ids(track_ids):
    tracks = (
        requests.post(f"{app_url}/tracks_without_mentioned_by", json=track_ids)
    ).json()
    return tracks


# TODO - sometimes PATCH gives nothing? empty recommendation?


def get_tracks_and_reactions_for_playlist(playlist_id):
    tracks = (requests.post(f"{app_url}/tracks_of_playlist", json=playlist_id)).json()
    return tracks


def enumerate_artist_id(tracks1, tracks2):
    # Combine unique artist IDs from both dictionaries
    unique_artist_ids = {track['artist_id'] for track in tracks1 + tracks2}

    artist_id_to_int = {
        artist_id: idx for idx, artist_id in enumerate(unique_artist_ids)
    }

    updated_tracks1 = [
        {
            **track,
            "artist_id": artist_id_to_int[track['artist_id']],
        }
        for track in tracks1
    ]
    
    updated_tracks2 = [
        {
            **track,
            "artist_id": artist_id_to_int[track['artist_id']],
        }
        for track in tracks2
    ]

    return updated_tracks1, updated_tracks2



class UpdateGroupReccomendations:
    def __init__(self, playlist_id):
        self.playlist_id = playlist_id
        self._batch_size = 10
        self.recommended_tracks = self.recommend_more()

    def get(self):
        return self.recommended_tracks

    def prepare_features(self, data):
        '''
        get features of given data tracks
        '''
        X = []

        for entry in data:
            features = [
                entry["acousticness"],
                entry["artist_id"],
                entry["danceability"],
                entry["duration_ms"],
                entry["energy"],
                entry["instrumentalness"],
                entry["key"],
                entry["liveness"],
                entry["loudness"],
                entry["popularity"],
                entry["release_date"],
                entry["speechiness"],
                entry["tempo"],
                entry["valence"],
            ]
            X.append(features)

        return np.array(X)

    def predict(self, data, test_data):
        '''
        method predicting if test_data would be classified as skip or add
        to the recommendations playlist using LinearSVC
        '''
        X = self.prepare_features(data)

        test_data = self.prepare_features(test_data)

        y = np.array([entry["reaction"] for entry in data])

        model = LinearSVC(max_iter=100)
        model.fit(X, y)

        predictions = model.predict(test_data)

        return predictions

    def recommend_more(self):
        '''
        main method creating tracks for class, connecting all methods
        '''
        data = get_tracks_and_reactions_for_playlist(self.playlist_id)  # TODO -

        track_ids = {track['track_id'] for track in data}
        track_ids = list(track_ids)

        propositions = get_tracks_without_mentioned_by_ids(track_ids)

        propositions, data = enumerate_artist_id(propositions, data)

        predictions = self.predict(data, propositions)

        recommended_tracks = [
            propositions[i]['track_id']
            for i in range(len(predictions))
            if predictions[i] == 1
        ]

        return recommended_tracks[: self._batch_size]
