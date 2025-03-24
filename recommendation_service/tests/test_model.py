import pytest
import numpy as np
from datetime import datetime, timedelta
from init_gen import GroupReccomendations
from sklearn.cluster import KMeans

@pytest.fixture
def group_rec():
    """ Przygotowanie instancji GroupReccomendations przed testami """
    user_ids = [101, 203, 306]
    return GroupReccomendations(user_ids)

def test_initialization(group_rec):
    """ Sprawdza, czy obiekt inicjalizuje się poprawnie """
    assert group_rec.user_ids == [101, 203, 306]
    assert isinstance(group_rec._time_window_start, datetime)
    assert isinstance(group_rec._time_window_end, datetime)

def test_prepare_features(group_rec):
    """ Sprawdza, czy funkcja poprawnie przetwarza dane wejściowe """
    test_tracks = [
        {"danceability": 0.5, "energy": 0.7, "loudness": -5, "speechiness": 0.1, 
         "instrumentalness": 0.0, "liveness": 0.3, "tempo": 120, "artist_id": 1},
        {"danceability": 0.8, "energy": 0.6, "loudness": -6, "speechiness": 0.05, 
         "instrumentalness": 0.1, "liveness": 0.4, "tempo": 130, "artist_id": 2}
    ]
    features = group_rec.prepare_features(test_tracks)
    assert features.shape == (2, 7)  # Powinny być 2 tracki i 7 cech

def test_encode_artist_ids(group_rec):
    """ Sprawdza, czy ID artystów są poprawnie kodowane """
    test_tracks = [
        {"track_id": "a1", "artist_id": "art1"},
        {"track_id": "a2", "artist_id": "art2"},
        {"track_id": "a3", "artist_id": "art1"}
    ]
    encoded_tracks = group_rec.encode_artist_ids(test_tracks)
    artist_ids = {track["artist_id"] for track in encoded_tracks}
    assert len(artist_ids) == 2  # Powinny być 2 unikalne ID

def test_get_weighed_tracks(group_rec):
    """ Sprawdza, czy funkcja zwraca słownik z wartościami w zakresie (-1, 1) """
    user_id = 1
    result = group_rec.get_weighed_tracks(user_id)
    assert isinstance(result, dict)
    assert all(-1 <= score <= 1 for score in result.values())

def test_create_recommendations_basic(group_rec):
    """ Sprawdza, czy metoda zwraca listę rekomendacji o poprawnej długości """
    group_rec.user_tracks = [
        {"track_id": f"track_{i}", "danceability": 0.5} for i in range(10)
    ]  # Symulacja danych testowych
    recommendations = group_rec.create_recommendations_basic()
    assert isinstance(recommendations, list)
    assert len(recommendations) == group_rec._final_playlist_length

def test_create_recommendations_advanced(group_rec):
    """ Sprawdza, czy metoda zaawansowana zwraca rekomendacje """
    group_rec.user_tracks = [
        {"track_id": f"track_{i}", "danceability": 0.5} for i in range(10)
    ]  # Symulacja danych testowych
    recommendations = group_rec.create_recommendations_advanced()
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0
