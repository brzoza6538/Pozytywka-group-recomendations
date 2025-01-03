import numpy as np


class PredictionModel:
    def __init__(self):
        self.weights = None

    def prepare_data_for_user(self, user_id, user_session, tracks):
        '''
        Przygotowuje dane treningowe dla użytkownika
        na podstawie historii sesji użytkownika oraz dodatkowych
        informacji o utworach.

        :param user_id: ID użytkownika
        :param user_session: Dane sesji użytkownika
        :param tracks: Dane utworów (track_id -> features)
        :return: Dane użytkownika (lista [cechy, akceptacja])
        '''
        # Filtrowanie sesji użytkownika
        user_sessions = user_session[user_session['user_id'] == user_id]

        # Tworzenie danych
        user_data = []
        for _, session in user_sessions.iterrows():
            if session['event_type'] == 'play' and session['track_id'] is not None:
                track_id = session['track_id']

                # Sprawdź, czy mamy dane o tym utworze
                if track_id in tracks:
                    track_info = tracks[track_id]
                    track_features = [
                        track_info['danceability'],
                        track_info['energy'],
                        track_info['popularity'],
                        track_info['duration_ms'],
                        track_info['explicit'],
                        track_info['loudness'],
                        track_info['speechiness'],
                        track_info['acousticness'],
                        track_info['instrumentalness'],
                        track_info['liveness'],
                        track_info['valence'],
                        track_info['tempo']
                    ]

                    # Akceptacja - 1 = play, 0 = skip
                    user_data.append((track_features, 1))  # 1 oznacza play

            elif session['event_type'] == 'skip' and session['track_id'] is not None:
                track_id = session['track_id']

                # Sprawdź, czy mamy dane o tym utworze
                if track_id in tracks:
                    track_info = tracks[track_id]
                    track_features = [
                        track_info['danceability'],
                        track_info['energy'],
                        track_info['popularity'],
                        track_info['duration_ms'],
                        track_info['explicit'],
                        track_info['loudness'],
                        track_info['speechiness'],
                        track_info['acousticness'],
                        track_info['instrumentalness'],
                        track_info['liveness'],
                        track_info['valence'],
                        track_info['tempo']
                    ]

                    # Akceptacja - 0 oznacza skip
                    user_data.append((track_features, 0))  # 0 oznacza skip

        return user_data

    def prepare_all_data(self, list_user_id, list_user_session, tracks):
        '''
        Sumuje dane treningowe każdego z użytkowników.

        :param list_user_id: Lista ID użytkowników
        :param list_user_session: Dane sesji wszystkich użytkowników
        :param tracks: Dane utworów (track_id -> features)
        :return: Połączone dane dla wszystkich użytkowników
        '''
        all_data = []
        for user_id in list_user_id:
            user_session = list_user_session[list_user_session['user_id'] ==
                                             user_id]
            user_data = self.prepare_data_for_user(user_id,
                                                   user_session,
                                                   tracks)
            all_data.extend(user_data)

        return all_data

    def train(self, X, y):
        '''
        Trenuje model na podstawie dostarczonych danych.

        :param X: Dane wejściowe (cechy)
        :param y: Dane wyjściowe (etykiety)
        '''
        X = np.hstack([np.ones((X.shape[0], 1)), X])  # Dodaj stałą do cech
        self.weights = np.linalg.pinv(X.T @ X) @ X.T @ y

    def predict(self, song_id, tracks, artists):
        '''
         Przewiduje ocenę dla piosenki na podstawie song_id.

        :param song_id: ID piosenki
        :param tracks: Dane utworów (track_id -> features)
        :param artists: Dane artystów (artist_id -> genres)
        :return: Wynik predykcji
        '''
        if self.weights is None:
            raise ValueError("Model nie został przeszkolony." +
                             "Użyj metody train().")

        # Sprawdź, czy mamy dane o tym utworze
        if song_id not in tracks:
            raise ValueError(f"Nie znaleziono piosenki o ID {song_id}")

        track_info = tracks[song_id]
        track_features = [
            track_info['danceability'],
            track_info['energy'],
            track_info['popularity'],
            track_info['duration_ms'],
            track_info['explicit'],
            track_info['loudness'],
            track_info['speechiness'],
            track_info['acousticness'],
            track_info['instrumentalness'],
            track_info['liveness'],
            track_info['valence'],
            track_info['tempo']
        ]

        features = np.hstack([1, track_features])

        prediction = features @ self.weights
        return 1 / (1 + np.exp(-prediction))  # Skalowanie do przedziału [0, 1]
