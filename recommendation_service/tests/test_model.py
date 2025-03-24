import unittest
import numpy as np
from datetime import datetime, timedelta
from init_gen import GroupReccomendations

class TestGroupRecommendations(unittest.TestCase):

    def setUp(self):
        """ Przygotowanie danych testowych przed każdym testem """
        self.user_ids = [1, 2, 3]  # Przykładowe ID użytkowników
        self.group_rec = GroupReccomendations(self.user_ids)

    def test_initialization(self):
        """ Sprawdza, czy obiekt inicjalizuje się poprawnie """
        self.assertEqual(self.group_rec.user_ids, [1, 2, 3])
        self.assertIsInstance(self.group_rec._time_window_start, datetime)
        self.assertIsInstance(self.group_rec._time_window_end, datetime)

    def test_prepare_features(self):
        """ Sprawdza, czy funkcja poprawnie przetwarza dane wejściowe """
        test_tracks = [
            {"danceability": 0.5, "energy": 0.7, "loudness": -5, "speechiness": 0.1, 
             "instrumentalness": 0.0, "liveness": 0.3, "tempo": 120, "artist_id": 1},
            {"danceability": 0.8, "energy": 0.6, "loudness": -6, "speechiness": 0.05, 
             "instrumentalness": 0.1, "liveness": 0.4, "tempo": 130, "artist_id": 2}
        ]
        features = self.group_rec.prepare_features(test_tracks)
        self.assertEqual(features.shape, (2, 7))  # Powinny być 2 tracki i 7 cech

    def test_encode_artist_ids(self):
        """ Sprawdza, czy ID artystów są poprawnie kodowane """
        test_tracks = [
            {"track_id": "a1", "artist_id": "art1"},
            {"track_id": "a2", "artist_id": "art2"},
            {"track_id": "a3", "artist_id": "art1"}
        ]
        encoded_tracks = self.group_rec.encode_artist_ids(test_tracks)
        artist_ids = {track["artist_id"] for track in encoded_tracks}
        self.assertEqual(len(artist_ids), 2)  # Powinny być 2 unikalne ID

    def test_get_weighed_tracks(self):
        """ Sprawdza, czy funkcja zwraca słownik z wartościami w zakresie (-1, 1) """
        user_id = 1
        result = self.group_rec.get_weighed_tracks(user_id)
        self.assertIsInstance(result, dict)
        self.assertTrue(all(-1 <= score <= 1 for score in result.values()))

    def test_cluster_tracks(self):
        """ Sprawdza, czy klastrowanie działa poprawnie """
        test_tracks = [
            {"danceability": 0.5, "energy": 0.7, "loudness": -5, "speechiness": 0.1, 
             "instrumentalness": 0.0, "liveness": 0.3, "tempo": 120, "artist_id": 1},
            {"danceability": 0.8, "energy": 0.6, "loudness": -6, "speechiness": 0.05, 
             "instrumentalness": 0.1, "liveness": 0.4, "tempo": 130, "artist_id": 2}
        ]
        clusters = self.group_rec.cluster_tracks(test_tracks, taste_groups=2)
        self.assertEqual(len(clusters), 2)  # Powinny być 2 grupy

    def test_create_recommendations_basic(self):
        """ Sprawdza, czy metoda zwraca listę rekomendacji o poprawnej długości """
        recommendations = self.group_rec.create_recommendations_basic()
        self.assertIsInstance(recommendations, list)
        self.assertEqual(len(recommendations), self.group_rec._final_playlist_length)

if __name__ == "__main__":
    unittest.main()
