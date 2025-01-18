from sklearn.svm import LinearSVC

import numpy as np
import requests
from datetime import datetime, timedelta

app_url = "http://app:8000"


def get_type_of_tracks(user_id, event_type, from_time, to_time=datetime.utcnow()):
    data = [user_id, event_type, from_time.isoformat(), to_time.isoformat()]
    user_records = (requests.post(f"{app_url}/users_actions_of_type", json=data)).json()
    return user_records

def get_tracks_by_ids(track_ids):
    tracks = (requests.post(f"{app_url}/track_by_id", json=track_ids)).json()
    return tracks


def get_tracks_without_mentioned_by_ids(track_ids):
    tracks = (
        requests.post(f"{app_url}/tracks_without_mentioned_by", json=track_ids)
    ).json()
    return tracks

def get_tracks_and_reactions_for_playlist(playlist_id):
    tracks = (requests.post(f"{app_url}/tracks_of_playlist", json=playlist_id)).json()
    return tracks


def enumerate_artist_id(tracks1, tracks2):
    """ 
    Combine unique artist IDs from both dictionaries
    """
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
        self.recommended_tracks = []

    def get(self):
        self.recommended_tracks = self.recommend_more()
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

    def predict(self, data, train_data):
        '''
        method predicting if test_data would be classified as skip or add
        to the recommendations playlist using LinearSVC
        '''
        X = self.prepare_features(data)

        train_data = self.prepare_features(train_data)

        y = np.array([entry["reaction"] for entry in data])

        model = LinearSVC(max_iter=100)
        model.fit(X, y)

        predictions = model.predict(train_data)

        return predictions

    def recommend_more(self):
        '''
        main method creating tracks for class, connecting all methods
        '''
        data = get_tracks_and_reactions_for_playlist(self.playlist_id)

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
    
    def get_user_data(self, user_id, time_window_start, time_window_end):
        liked_data = get_type_of_tracks(user_id, "like", time_window_start, time_window_end)
        skipped_data = get_type_of_tracks(user_id, "skip", time_window_start, time_window_end)
        records = {}

        for track_id in set(liked_data.keys()).union(skipped_data.keys()):
            records[track_id] = (liked_data.get(track_id, 0) - skipped_data.get(track_id, 0))

        data = get_tracks_by_ids(list(records.keys()))

        for record in data:
            record["reaction"] = (False if records[record["track_id"]] < 0 else True)

        return data

    def test_recommendation(self, user_ids):
        """
            test algorithm training it on users's history of last year, test based on last 3 months 
        """
        time_window_start = datetime.utcnow() - timedelta(days=360)
        time_window_end = datetime.utcnow()- timedelta(days=90)

        results = []
        accuracy_counter = 0 
        d_counter = 0 

        for user_id in user_ids:
            data = self.get_user_data(user_id, time_window_start, time_window_end) [:30]

            track_ids = {track['track_id'] for track in data}
            track_ids = list(track_ids)
            
            propositions =  self.get_user_data(user_id, time_window_end, datetime.utcnow())
            propositions = [{key: value for key, value in track.items() if key != "reaction"} for track in propositions]

            propositions, data = enumerate_artist_id(propositions, data)
            predictions = self.predict(data, propositions)

            test_data = self.get_user_data(user_id, time_window_end, datetime.utcnow())
            accuracy_counter = 0 
            d_counter = 0 

            for i in range(len(predictions)):
                if (predictions[i] == 1 and test_data[i]['reaction'] == True):
                    accuracy_counter += 1
                else:
                    d_counter += 1
            print(round(accuracy_counter/len(test_data), 4))
            results.append(f"{user_id}   {round(accuracy_counter/len(test_data), 4)}")
        return results