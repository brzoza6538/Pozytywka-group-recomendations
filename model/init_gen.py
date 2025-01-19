import itertools
import random
from datetime import datetime, timedelta

import numpy as np
import requests
from requests_to_app import (get_tracks_by_ids,
                                    get_tracks_without_mentioned_by_ids,
                                    get_type_of_tracks)
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.tree import DecisionTreeRegressor


class GroupReccomendations:
    def __init__(self, user_ids):

        self.user_ids = user_ids

        self._time_window_start = datetime.utcnow() - timedelta(days=180)
        self._time_window_end = datetime.utcnow()

        self._users_favourite_tracks_amount = 50 * len(self.user_ids)
        self._cluster_recommendation = 10
        # powinno być zależne od ilości osób czy nie?
        self._taste_groups = 5 * len(self.user_ids)

        self._liked_weight = 5
        self._skipped_weight = -3
        self._started_weight = 3

        self.normalisation_range_up = 1
        self.normalisation_range_down = -1

        self._final_playlist_length = 30

        self._used_features = [
            "danceability",
            "energy",
            "loudness",
            "speechiness",
            #  "acousticness",
            "instrumentalness",
            "liveness",
            #  "valence",
            "tempo"
        ]

        self.recommendations = []

    def get_advanced(self):
        self.recommendations = self.create_recommendations_advanced()
        return self.recommendations

    def get_basic(self):
        self.recommendations = self.create_recommendations_basic()
        return self.recommendations

    def prepare_features(self, tracks_data, discrete=False):
        '''
        Get features of tracks, optionally including artist_id.
        '''
        features = []
        for track in tracks_data:
            feature_vector = [
                track[feature] for feature in self._used_features
            ]
            if discrete:
                feature_vector.append(track["artist_id"])
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

            started_tracks[track_id] = self.normalisation_range_down + (
                self.normalisation_range_up - self.normalisation_range_down) * (1 / (1 + np.exp(-record)))
        return started_tracks

    def get_top_tracks(self, limit=True):
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
        if limit:
            top_tracks = [track_id for track_id,
                          score in sorted_items[: self._users_favourite_tracks_amount]]
        else:
            top_tracks = [track_id for track_id, score in sorted_items]

        return top_tracks

    def cluster_tracks(self, tracks_data, taste_groups=None):
        '''
        group most liked music of users using Kmeans
        '''
        if taste_groups is None:
            taste_groups = self._taste_groups

        tracks_data = self.encode_artist_ids(tracks_data)

        features = self.prepare_features(tracks_data, discrete=True)

        kmeans = KMeans(n_clusters=taste_groups)
        clusters = kmeans.fit_predict(features)

        result = [[] for _ in range(taste_groups)]

        for i, track in enumerate(tracks_data):
            result[clusters[i]].append(track)

        return result

    def recommend_tracks_for_cluster(self, tracks_data):
        '''
        recommend tracks for each individual cluster based on how "close" they are to the average of a group
        '''
        liked_tracks_ids = [track["track_id"] for track in tracks_data]

        # not enough data to use the whole database
        # propositions_pool = get_tracks_without_mentioned_by_ids(liked_tracks_ids)
        propositions_pool = get_tracks_by_ids(self.get_top_tracks(limit=False))

        liked_tracks_features = self.prepare_features(tracks_data)
        propositions_features = self.prepare_features(propositions_pool)

        average_features = np.mean(liked_tracks_features, axis=0)

        similarities = cosine_similarity(
            propositions_features, [average_features])

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
        features_list = self.prepare_features(train_tracks)
        labels = list(train_data.values())

        tree = DecisionTreeRegressor(max_depth=g_max_depth)
        tree.fit(features_list, labels)

        test_tracks = get_tracks_by_ids(test_data)
        test_features = self.prepare_features(test_tracks)
        predictions = {}
        for track, features in zip(test_tracks, test_features):
            score = tree.predict([features])[0]
            predictions[track["track_id"]] = max(0, min(1, score))
        return predictions

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
            # recommend tracks for each individual cluster
            recommendations += self.recommend_tracks_for_cluster(cluster)

        recommendations = random.sample(
            recommendations, self._final_playlist_length)
        # instead of choosing the ones with best predicted weight, just choose random tracks from clusters
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
            # recommend tracks for each individual cluster
            recommendations += self.recommend_tracks_for_cluster(cluster)

        data = {}
        for user_id in self.user_ids:
            predictions = self.evaluate_tracks(
                self.get_weighed_tracks(user_id), recommendations
            )
            data = {
                key: data.get(key, 0) + predictions.get(key, 0)
                for key in data | predictions
            }  # predict score of each recommended track for each user using decision tree

        recommendations = sorted(data, key=data.get, reverse=True)[
            : self._final_playlist_length]

        return recommendations
