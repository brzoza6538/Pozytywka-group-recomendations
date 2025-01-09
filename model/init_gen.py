from flask import Blueprint, request, render_template
from models import Recommendation, User, Track, Session, db
from datetime import datetime, timedelta

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

import random


def get_tracks_by_ids(track_ids):
   tracks = []
   for track_id in track_ids:
      tracks.append((Track.query.filter(Track.track_id == track_id).first()).to_dict())
   return tracks

def get_tracks_without_mentioned_by_ids(track_ids):
    tracks = Track.query.filter(~Track.track_id.in_(track_ids)).all()
    return [track.to_dict() for track in tracks]

def get_type_of_tracks(user_id, time_border, event_type):
   sessions = (
      Session.query
      .filter(
         (Session.user_id == user_id) & 
         (Session.event_type == event_type) &
         (Session.timestamp < time_border)
      )
      .all()
   )

   user_records = {}

   for session in sessions:
      if (session.track_id not in user_records.keys()):
         user_records[session.track_id] = 0
      user_records[session.track_id] += 1
   
   return user_records


def prepare_features(tracks_data):
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
         track["artist_id"]
      ]
      features.append(feature_vector)
   return np.array(features)

def prepare_features_without_discrete(tracks_data):
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

def encode_artist_ids(tracks_data):
   '''
   turn artist_id into unique for given set int value
   '''
   unique_artist_ids = {track['artist_id'] for track in tracks_data}
   
   artist_id_to_int = {artist_id: idx for idx, artist_id in enumerate(unique_artist_ids)}
   
   for track in tracks_data:
      track['artist_id'] = artist_id_to_int[track['artist_id']]
   
   return tracks_data



class GroupReccomendations:
   def __init__(self, user_ids):
      self.TIME_BORDER = datetime.utcnow() - timedelta(days=90)
      self.USERS_LIKED_TRACKS_AMOUNT = 100
      self.CLUSTER_RECOMMENDATION = 10
      self.TASTE_GROUPS = 5 #TODO zależne od ilości osób? 

      self.LIKED_WEIGHT = 5
      self.SKIPPED_WEIGHT = -5
      self.STARTED_WEIGHT = 4

      self.FINAL_PLAYLIST_LENGTH = 30

      self.user_ids = user_ids
      self.create_recommendations()


   def get_weighed_tracks(self, user_id):
      '''
      score songs of a user with given user_id
      '''
      started_tracks = get_type_of_tracks(user_id, self.TIME_BORDER, "play")
      skipped_tracks = get_type_of_tracks(user_id, self.TIME_BORDER, "skip")
      liked_tracks = get_type_of_tracks(user_id, self.TIME_BORDER, "like")

      for track_id in started_tracks.keys():
         record = (
            started_tracks.get(track_id, 0) * self.STARTED_WEIGHT +
            skipped_tracks.get(track_id, 0) * self.SKIPPED_WEIGHT +
            liked_tracks.get(track_id, 0) * self.LIKED_WEIGHT 
         )
         started_tracks[track_id] = 1 / (1 + np.exp(-record))

      return started_tracks

   def get_top_tracks(self):
      '''
      get best regarded songs of users
      '''
      connected_scores = {} 

      for user_id in self.user_ids:
         user_scores = self.get_weighed_tracks(user_id)

         for track_id in user_scores.keys():
            if(track_id not in connected_scores.keys()):
               connected_scores[track_id] = user_scores[track_id]
            else:
               connected_scores[track_id] += user_scores[track_id]

      sorted_items = sorted(connected_scores.items(), key=lambda item: item[1], reverse=True)
      top_tracks = [track_id for track_id, score in sorted_items[:self.USERS_LIKED_TRACKS_AMOUNT]]
      return top_tracks


   def cluster_tracks(self, tracks_data):
      '''
      group most liked music of users using Kmeans
      '''
      tracks_data = encode_artist_ids(tracks_data)

      features = prepare_features(tracks_data)

      kmeans = KMeans(n_clusters=self.TASTE_GROUPS)
      clusters = kmeans.fit_predict(features)


      result = [[] for _ in range(self.TASTE_GROUPS)]  

      for i, track in enumerate(tracks_data):
         result[clusters[i]].append(track)

      return result


   def recommend_tracks_for_cluster(self, tracks_data):
      '''
      recommend tracks for each individual cluster
      '''
      liked_tracks_ids = [track["track_id"] for track in tracks_data]
      propositions_pool = get_tracks_without_mentioned_by_ids(liked_tracks_ids)
      
      liked_tracks_features = prepare_features_without_discrete(tracks_data)
      propositions_features = prepare_features_without_discrete(propositions_pool)

      average_features = np.mean(liked_tracks_features, axis=0)

      similarities = cosine_similarity(propositions_features, [average_features])
      
      propositions_with_similarity = [(track, similarity[0]) for track, similarity in zip(propositions_pool, similarities)]
      propositions_with_similarity.sort(key=lambda x: x[1], reverse=True)
      recommended_tracks = [track["track_id"] for track, _ in propositions_with_similarity[:self.CLUSTER_RECOMMENDATION]]
      
      return recommended_tracks

   def evaluate_tracks(self, train_data, test_data):
      '''
      predict score of songs in test_data based on weighted train_data using SVM
      '''
      train_tracks = [get_tracks_by_ids([track_id])[0] for track_id in train_data.keys()]
      features_list = prepare_features_without_discrete(train_tracks)
      labels = list(train_data.values())

      tree = DecisionTreeRegressor(max_depth=5)
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
            track["tempo"]
         ]
         score = tree.predict([track_features])[0]
         predictions[track["track_id"]] = max(0, min(1, score))

      return predictions
      

   def test_tree_accuracy(self):
      '''
      TODO - move somewhere else 

      used to test accuracy of SVM based scores 
      '''
      users_id = [101,
      202, 303, 404, 505, 606, 707, 808, 909,
      102, 202, 302, 402, 502, 602, 702, 802, 902
      ]


      diff = []

      results  = ""

      for user_id in users_id:

         train_data = self.get_weighed_tracks(user_id)


         train_data = dict(list(train_data.items())[20:])
         test_data = dict(list(train_data.items())[:20])


         tracks_data = [track_id for track_id in test_data.keys()]

         prediction = self.evaluate_tracks(train_data, test_data)

         for track in tracks_data:
            results += f" {track}  : {round(prediction[track], 3)} : {round(test_data.get(track, 0), 3)} ---- {round(abs(prediction[track] - test_data.get(track, 0)), 3)}\n"
            diff.append(abs(prediction[track] - test_data.get(track, 0)))
      results += f"\n\n\-------\n\n min :  {round(np.min(diff), 3)}\n max : {round(np.max(diff), 3)}\n ddelt ---- {round(np.mean(diff), 3)}\n\n\-------\n\n"
      return str(results)



   def create_recommendations(self):
      '''
      main function creating tracks for class 
      '''

      tracks_ids = self.get_top_tracks()
      tracks_data = get_tracks_by_ids(tracks_ids)

      track_clusters = self.cluster_tracks(tracks_data) # group most liked music of users using Kmeans

      recommendations = []

      for cluster in track_clusters:
         recommendations += self.recommend_tracks_for_cluster(cluster) # recommend tracks for each individual cluster
      
      data = {}
      for user_id in self.user_ids:
         predictions = self.evaluate_tracks(self.get_weighed_tracks(user_id), recommendations)
         data = {key: data.get(key, 0) + predictions.get(key, 0) for key in data | predictions} # predict score of each recommended track for each user using SVM

      recommendation = sorted(data, key=data.get, reverse=True)[:self.FINAL_PLAYLIST_LENGTH] 

      self.recommendations = recommendation
