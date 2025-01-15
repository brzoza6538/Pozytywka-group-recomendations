import numpy as np
from models import Recommendation, Artist, User, Track, Session, db
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler


# TODO - sometimes PATCH gives nothing? empty recommendation?

def get_tracks_without_mentioned_by_ids(track_ids):
    tracks = Track.query.filter(~Track.track_id.in_(track_ids)).all()

    unique_artist_ids = {track.artist_id for track in tracks}
    artist_id_to_int = {artist_id: idx for idx, artist_id in enumerate(unique_artist_ids)}

    return [{
        **track.to_dict(),
        "artist_id": artist_id_to_int[track.artist_id],
    } for track in tracks]

def get_tracks_and_reactions_for_playlist(playlist_id):
    recommendations = (
        db.session.query(Recommendation, Track, Artist)
        .join(Track, Track.track_id == Recommendation.track_id)
        .join(Artist, Artist.id == Track.artist_id)
        .filter(Recommendation.playlist_id == playlist_id)
        .order_by(Recommendation.id.desc())
        .all()
    )

    unique_artist_ids = {track.artist_id for _, track, _ in recommendations}
    artist_id_to_int = {artist_id: idx for idx, artist_id in enumerate(unique_artist_ids)}

    return [
        {
            **track.to_dict(),
            "artist_id": artist_id_to_int[track.artist_id],  
            "reaction": int(recommendation.reaction)  
        }
        for recommendation, track, _ in recommendations
    ]



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
                entry["acousticness"], entry["artist_id"], entry["danceability"], entry["duration_ms"], entry["energy"], 
                entry["instrumentalness"], entry["key"], entry["liveness"], entry["loudness"], entry["popularity"], 
                entry["release_date"], entry["speechiness"], entry["tempo"], entry["valence"]
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
        data = get_tracks_and_reactions_for_playlist(self.playlist_id)

        track_ids = {track['track_id'] for track in data}
        track_ids = list(track_ids)

        propositions = get_tracks_without_mentioned_by_ids(track_ids)

        predictions = self.predict(data, propositions)

        recommended_tracks = [
            propositions[i]['track_id'] for i in range(len(predictions)) if predictions[i] == 1
        ]

        return recommended_tracks[:self._batch_size]