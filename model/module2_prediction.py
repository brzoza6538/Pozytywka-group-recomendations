import numpy as np


class PredictionModel:
    def __init__(self):
        self.weights = None

    def prepare_data(self, session_data, song_data, user_id):
        '''
        Przygotowuje dane treningowe na podstawie historii sesji użytkownika.

        :param session_data: Dane sesji użytkownika
        :param song_data: Dane o piosenkach
        :param user_id: ID użytkownika
        :return: Dane treningowe (X, y)
        '''
        user_sessions = session_data[session_data['user_id'] == user_id]
        merged_data = user_sessions.merge(song_data, on='song_id', how='left')

        X = merged_data.drop(columns=['acceptance', 'user_id', 'song_id'])
        y = merged_data['acceptance']  # 0 = skip, 1 = play/like

        return X.values, y.values

    def train(self, X, y):
        '''
        Trenuje model poprzez dopasowanie wag cech na podstawie prostego
        podejścia regresyjnego.

        :param X: Dane wejściowe (macierz cech)
        :param y: Etykiety (0 lub 1)
        '''
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        # Obliczanie wag za pomocą regresji liniowej (normal equation)
        self.weights = np.linalg.pinv(X.T @ X) @ X.T @ y

    def predict(self, song_features):
        '''
        Przewiduje akceptację piosenki na podstawie jej cech.

        :param song_features: Dane wejściowe dla pojedynczej piosenki
        :return: Wartość przewidywania (float)
        '''
        if self.weights is None:
            raise ValueError("Model nie został przeszkolony. " +
                             "Użyj metody train().")

        song_features = np.hstack([1, song_features])
        prediction = song_features @ self.weights

        # Skalowanie wyniku do przedziału [0, 1]
        return 1 / (1 + np.exp(-prediction))
