from datetime import datetime, timedelta

from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import itertools
import requests
import random

app_url = "http://app:8000"


def get_tracks_by_ids(track_ids):
    tracks = (requests.post(f"{app_url}/track_by_id", json=track_ids)).json()
    return tracks


def get_tracks_without_mentioned_by_ids(track_ids):
    tracks = (
        requests.post(f"{app_url}/tracks_without_mentioned_by", json=track_ids)
    ).json()
    return tracks


def get_type_of_tracks(user_id, event_type, from_time, to_time=datetime.utcnow()):
    data = [user_id, event_type, from_time.isoformat(), to_time.isoformat()]
    user_records = (requests.post(f"{app_url}/users_actions_of_type", json=data)).json()
    return user_records


class GroupReccomendations:
    def __init__(self, user_ids):
        self._time_window_start = datetime.utcnow() - timedelta(days=180)
        self._time_window_end = datetime.utcnow()

        self._users_favourite_tracks_amount = 70  * len(user_ids)
        self._cluster_recommendation = 30
        self._taste_groups = 5  * len(user_ids) # powinno być zależne od ilości osób czy nie?

        self._liked_weight = 5
        self._skipped_weight = -5
        self._started_weight = 5

        self.normalisation_range_up = 1
        self.normalisation_range_down = -1

        self._final_playlist_length = 30

        self.user_ids = user_ids


        self.recommendations = []

    def get_advanced(self):
        self.recommendations = self.create_recommendations_advanced()
        return self.recommendations

    def get_basic(self):
        self.recommendations = self.create_recommendations_basic()
        return self.recommendations

    def prepare_features(self, tracks_data):
        '''
        get features of tracks with artist_id
        '''
        features = []
        for track in tracks_data:
            feature_vector = [
                track["danceability"],
                track["energy"],
                track["loudness"],
                track["speechiness"],
                track["acousticness"],
                track["instrumentalness"],
                track["liveness"],
                track["valence"],
                track["tempo"],
                track["artist_id"],
            ]
            features.append(feature_vector)
        return np.array(features)

    def prepare_features_without_discrete(self, tracks_data):
        '''
        get features of tracks without artist_id or other discrete columns
        '''
        features = []
        for track in tracks_data:
            feature_vector = [
                track["danceability"],
                track["energy"],
                track["loudness"],
                track["speechiness"],
                track["acousticness"],
                track["instrumentalness"],
                track["liveness"],
                track["valence"],
                track["tempo"],
            ]
            features.append(feature_vector)
        return np.array(features)

    def encode_artist_ids(self, tracks_data):
        '''
        turn artist_id into unique for given set int value
        '''
        unique_artist_ids = {track['artist_id'] for track in tracks_data}

        artist_id_to_int = {
            artist_id: idx for idx, artist_id in enumerate(unique_artist_ids)
        }

        for track in tracks_data:
            track['artist_id'] = artist_id_to_int[track['artist_id']]

        return tracks_data

    def get_weighed_tracks(self, user_id):
        '''
        score songs of a user with given user_id
        '''
        started_tracks = get_type_of_tracks(
            user_id, "play", self._time_window_start, self._time_window_end
        )
        skipped_tracks = get_type_of_tracks(
            user_id, "skip", self._time_window_start, self._time_window_end
        )
        liked_tracks = get_type_of_tracks(
            user_id, "like", self._time_window_start, self._time_window_end
        )

        for track_id in started_tracks.keys():
            record = (
                started_tracks.get(track_id, 0) * self._started_weight
                + skipped_tracks.get(track_id, 0) * self._skipped_weight
                + liked_tracks.get(track_id, 0) * self._liked_weight
            )

            started_tracks[track_id] = self.normalisation_range_down + (self.normalisation_range_up - self.normalisation_range_down) * (1 / (1 + np.exp(-record)))
        return started_tracks

    def get_top_tracks(self):
        '''
        get best regarded songs of users
        '''
        connected_scores = {}

        for user_id in self.user_ids:
            user_scores = self.get_weighed_tracks(user_id)

            for track_id in user_scores.keys():
                if track_id not in connected_scores.keys():
                    connected_scores[track_id] = user_scores[track_id]
                else:
                    connected_scores[track_id] += user_scores[track_id]

        sorted_items = sorted(
            connected_scores.items(), key=lambda item: item[1], reverse=True
        )
        top_tracks = [track_id for track_id, score in sorted_items[: self._users_favourite_tracks_amount]]


        return top_tracks

    def cluster_tracks(self, tracks_data, taste_groups=None):
        '''
        group most liked music of users using Kmeans
        '''
        if taste_groups is None:
            taste_groups = self._taste_groups

        tracks_data = self.encode_artist_ids(tracks_data)

        features = self.prepare_features(tracks_data)

        kmeans = KMeans(n_clusters=taste_groups)
        clusters = kmeans.fit_predict(features)

        result = [[] for _ in range(taste_groups)]

        for i, track in enumerate(tracks_data):
            result[clusters[i]].append(track)

        return result

    def test_clusters(self): 
        # to chyba nic nie mówi - to wynika z braku danych przy za dużej ilośći klastrów niż czegokolwiek innego
        tracks_ids = self.get_top_tracks()
        tracks_data = get_tracks_by_ids(tracks_ids)

        accuracy = []
        for i in [2, 8, 32, 48, 128]:
            track_clusters = self.cluster_tracks(tracks_data, i)

            X = []
            y = []

            for cluster_idx, cluster in enumerate(track_clusters):
                for track in cluster:
                    features = [
                        track["danceability"],
                        track["energy"],
                        track["loudness"],
                        track["speechiness"],
                        track["acousticness"],
                        track["instrumentalness"],
                        track["liveness"],
                        track["valence"],
                        track["tempo"],
                    ]
                    X.append(features)
                    y.append(cluster_idx)

            X = np.array(X)
            y = np.array(y)

            model = RandomForestClassifier(n_estimators=100)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy.append(accuracy_score(y_test, y_pred))

            print(f"{i} -- {accuracy}")
        return str(accuracy)


    def recommend_tracks_for_cluster(self, tracks_data):
        '''
        recommend tracks for each individual cluster based on how "close" they are to the average of a group
        '''
        liked_tracks_ids = [track["track_id"] for track in tracks_data]

        # propositions_pool = get_tracks_without_mentioned_by_ids(liked_tracks_ids)
        propositions_pool = get_tracks_by_ids(self.get_top_tracks())

        liked_tracks_features = self.prepare_features_without_discrete(tracks_data)
        propositions_features = self.prepare_features_without_discrete(propositions_pool)

        average_features = np.mean(liked_tracks_features, axis=0)

        similarities = cosine_similarity(propositions_features, [average_features])

        propositions_with_similarity = [
            (track, similarity[0])
            for track, similarity in zip(propositions_pool, similarities)
        ]
        propositions_with_similarity.sort(key=lambda x: x[1], reverse=True)
        recommended_tracks = [
            track["track_id"]
            for track, _ in propositions_with_similarity[: self._cluster_recommendation]
        ]

        return recommended_tracks

    def evaluate_tracks(self, train_data, test_data, g_max_depth=10):
        '''
        predict score of songs in test_data based on weighted train_data using decision tree
        '''
        train_tracks = [
            get_tracks_by_ids([track_id])[0] for track_id in train_data.keys()
        ]
        features_list = self.prepare_features_without_discrete(train_tracks)
        labels = list(train_data.values())

        tree = DecisionTreeRegressor(max_depth=g_max_depth)
        tree.fit(features_list, labels)

        test_tracks = get_tracks_by_ids(test_data)
        predictions = {}

        for track in test_tracks:
            track_features = [
                track["danceability"],
                track["energy"],
                track["loudness"],
                track["speechiness"],
                track["acousticness"],
                track["instrumentalness"],
                track["liveness"],
                track["valence"],
                track["tempo"],
            ]
            score = tree.predict([track_features])[0]
            predictions[track["track_id"]] = max(0, min(1, score))

        return predictions

    def test_tree_accuracy(self):
        '''
        TODO - move somewhere else

        used to test accuracy of decision tree based scores
        '''
        liked_weight_p = [1, 5]
        skipped_weight_p = [-5, -10]
        started_weight_p = [1, 3]
        depths = [10]
        normalisation_range_up_p = [1]
        normalisation_range_down_p = [0]
      
        constraints = list(
            itertools.product(
                liked_weight_p,
                skipped_weight_p,
                started_weight_p,
                depths,
                normalisation_range_up_p,
                normalisation_range_down_p
            )
        )
        results = []

        for setup in constraints:
            self._liked_weight = setup[0]
            self._skipped_weight = setup[1]
            self._started_weight = setup[2]

            self.normalisation_range_up = setup[4]
            self.normalisation_range_down = setup[5]
            diff = []
            for user_id in self.user_ids:
                train_data = self.get_weighed_tracks(user_id)

                train_data = list(train_data.items())

                train_items, test_items = train_test_split(train_data, test_size=0.2, random_state=42)

                train_data = dict(train_items)
                test_data = dict(test_items)

                tracks_data = [track_id for track_id in test_data.keys()]

                prediction = self.evaluate_tracks(train_data, test_data, setup[3])

                for track in tracks_data:
                    diff.append(abs(prediction[track] - test_data.get(track, 0)))
            results.append(f"\nmin :  {round(np.min(diff), 3)} max : {round(np.max(diff), 3)} ddelt : {round(np.mean(diff), 3)} \nsetup : {setup}\n")
            print(f"\nmin :  {round(np.min(diff), 3)} max : {round(np.max(diff), 3)} ddelt : {round(np.mean(diff), 3)} \nsetup : {setup}\n")
        return results


    def create_recommendations_basic(self):
        '''
        main method creating  tracks for class using basic version
        '''

        tracks_ids = self.get_top_tracks()
        tracks_data = get_tracks_by_ids(tracks_ids)

        track_clusters = self.cluster_tracks(
            tracks_data
        )  # group most liked music of users using Kmeans

        recommendations = []

        for cluster in track_clusters:
            recommendations += self.recommend_tracks_for_cluster(cluster)  # recommend tracks for each individual cluster

        recommendations = random.sample(recommendations, self._final_playlist_length)
        #instead of choosing the ones with best predicted weight, just choose random tracks from clusters
        return recommendations


    def create_recommendations_advanced(self):
        '''
        main method creating tracks for class, connecting all methods
        '''

        tracks_ids = self.get_top_tracks()
        tracks_data = get_tracks_by_ids(tracks_ids)

        track_clusters = self.cluster_tracks(
            tracks_data
        )  # group most liked music of users using Kmeans

        recommendations = []

        for cluster in track_clusters:
            recommendations += self.recommend_tracks_for_cluster(cluster)  # recommend tracks for each individual cluster

        data = {}
        for user_id in self.user_ids:
            predictions = self.evaluate_tracks(
                self.get_weighed_tracks(user_id), recommendations
            )
            data = {
                key: data.get(key, 0) + predictions.get(key, 0)
                for key in data | predictions
            }  # predict score of each recommended track for each user using decision tree

        recommendations = sorted(data, key=data.get, reverse=True)[: self._final_playlist_length]

        return recommendations

    def test_create_recommendations_advanced(self):
        time_start_p = [180, 360, 720]
        time_end_p = [90]
        users_favourite_tracks_amount_p = [70]
        cluster_recommendation_p = [30]
        taste_groups_p = [10]

        liked_weight_p = [5]
        skipped_weight_p = [-5]
        started_weight_p = [5]

        self.normalisation_range_up = [1]
        self.normalisation_range_down = [0]

        constraints = list(
            itertools.product(
                time_start_p,
                time_end_p,
                users_favourite_tracks_amount_p,
                cluster_recommendation_p,
                taste_groups_p,

                liked_weight_p,
                skipped_weight_p,
                started_weight_p,
                self.normalisation_range_up,
                self.normalisation_range_down
            )
        )
        results = []

        for setup in constraints:
            self._time_window_start = datetime.utcnow() - timedelta(days=setup[0])
            self._time_window_end = datetime.utcnow() - timedelta(days=setup[1])

            self._users_favourite_tracks_amount = setup[2] * len(self.user_ids)
            self._cluster_recommendation = setup[3]
            self._taste_groups = setup[4] * len(self.user_ids)

            self._liked_weight = setup[5]
            self._skipped_weight = setup[6]
            self._started_weight = setup[7]

            self.normalisation_range_up = setup[8]
            self.normalisation_range_down = setup[9]

            start = datetime.utcnow()
            recommendations = self.create_recommendations_advanced()
            end = datetime.utcnow()

            self._time_window_start = datetime.utcnow() - timedelta(days=setup[1])
            self._time_window_end = datetime.utcnow()
            test_tracks = self.get_top_tracks()

            duplicates = 0
            for track in recommendations:
                if track in test_tracks:
                    duplicates += 1
            results.append(
                f"\n{setup}\n   -  {len(test_tracks)}   - {duplicates} : {end - start}\n"
            )
            print(
                f"\n{setup}\n   -  {len(test_tracks)}   - {duplicates} : {(end - start).total_seconds()}\n"
            )

        return results
