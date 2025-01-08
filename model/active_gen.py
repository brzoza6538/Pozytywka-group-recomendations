import numpy as np
from models import Recommendation, Artist, User, Track, Session, db
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

BATCH_SIZE = 10

def get_tracks_without_mentioned_by_ids(track_ids):
    tracks = Track.query.filter(~Track.track_id.in_(track_ids)).all()

    unique_artist_ids = {track.artist_id for track in tracks}
    artist_id_to_int = {artist_id: idx for idx, artist_id in enumerate(unique_artist_ids)}

    return [{
        **track.to_dict(),
        "artist_id": artist_id_to_int[track.artist_id],  # Konwersja artist_id na int
    } for track in tracks]

def get_tracks_and_reactions_for_playlist(playlist_id):
    # Pobieramy rekomendacje, utwory i artystów
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

def prepare_data_for_prediction(data):
    X = []
    
    for entry in data:
        features = [
            entry["acousticness"], entry["artist_id"], entry["danceability"], entry["duration_ms"], entry["energy"], 
            entry["instrumentalness"], entry["key"], entry["liveness"], entry["loudness"], entry["popularity"], 
            entry["release_date"], entry["speechiness"], entry["tempo"], entry["valence"]
        ]
        X.append(features)
    
    return np.array(X)

def predict(data, test_data):
    X = prepare_data_for_prediction(data)

    test_data = prepare_data_for_prediction(test_data)

    y = np.array([entry["reaction"] for entry in data])

    model = LinearSVC(max_iter=100)
    model.fit(X, y)

    predictions = model.predict(test_data)

    return predictions



def recommend_more(playlist_id):
    data = get_tracks_and_reactions_for_playlist(playlist_id)

    track_ids = {track['track_id'] for track in data}
    track_ids = list(track_ids)

    propositions = get_tracks_without_mentioned_by_ids(track_ids)

    # Zwracamy prognozy

    predictions = predict(data, propositions)

    recommended_tracks = [
        propositions[i]['track_id'] for i in range(len(predictions)) if predictions[i] == 1
    ]

    return recommended_tracks[:BATCH_SIZE]
