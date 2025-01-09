from flask import Blueprint, request, render_template
from models import Recommendation, Artist, User, Track, Session, db
from datetime import datetime, timedelta

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

import random


TIME_BORDER = datetime.utcnow() - timedelta(days=90)
USERS_LIKED_TRACKS_AMOUNT = 100
CLUSTER_RECOMMENDATION = 10
TASTE_GROUPS = 5 #TODO zależne od ilości osób? 

LIKE_WEIGHT = 5
SKIP_WEIGHT = -5
PLAY_WEIGHT = 4

FINAL_PLAYLIST_LENGTH = 30

#znajdujesz z tego piosenki które wszystkim się podobają więc może być ich od cholery
#dopiero później skróć do n najlepiej ocenianych gdzie n jest na tyle małe że generowanie na bieżąco biędzie miało czas zadzialać


def get_liked_tracks(user_id):
   sessions = (
      Session.query
      .filter(
         (Session.user_id == user_id) & 
         (Session.event_type == "like") &
         (Session.timestamp < TIME_BORDER)
      )
      .all()
   )

   user_records = {}

   for session in sessions:
      if (session.track_id not in user_records.keys()):
         user_records[session.track_id] = 0
      user_records[session.track_id] += 1
   
   return user_records

def get_skipped_tracks(user_id):
   sessions = (
      Session.query
      .filter(
         (Session.user_id == user_id) & 
         (Session.event_type == "skip") &
         (Session.timestamp < TIME_BORDER)
      )
      .all()
   )

   user_records = {}

   for session in sessions:
      if (session.track_id not in user_records.keys()):
         user_records[session.track_id] = 0
      user_records[session.track_id] += 1
   
   return user_records



def get_started_tracks(user_id):
   sessions = (
      Session.query
      .filter(
         (Session.user_id == user_id) & 
         (Session.event_type == "play") &
         (Session.timestamp < TIME_BORDER)
      )
      .all()
   )

   user_records = {}

   for session in sessions:
      if (session.track_id not in user_records.keys()):
         user_records[session.track_id] = 0
      user_records[session.track_id] += 1
   
   return user_records



def get_weighed_tracks(user_id):
   started_tracks = get_started_tracks(user_id)
   skipped_tracks = get_skipped_tracks(user_id)
   liked_tracks = get_liked_tracks(user_id)

   for track_id in started_tracks.keys():
      record = (
         started_tracks.get(track_id, 0) * PLAY_WEIGHT +
         skipped_tracks.get(track_id, 0) * SKIP_WEIGHT +
         liked_tracks.get(track_id, 0) * LIKE_WEIGHT 
      )
      started_tracks[track_id] = 1 / (1 + np.exp(-record))

   return started_tracks


def get_top_tracks(user_ids):
   connected_scores = {} 

   for user_id in user_ids:
      user_scores = get_weighed_tracks(user_id)

      for track_id in user_scores.keys():
         if(track_id not in connected_scores.keys()):
            connected_scores[track_id] = user_scores[track_id]
         else:
            connected_scores[track_id] += user_scores[track_id]

   sorted_items = sorted(connected_scores.items(), key=lambda item: item[1], reverse=True)
   top_tracks = [track_id for track_id, score in sorted_items[:USERS_LIKED_TRACKS_AMOUNT]]
   return top_tracks


def get_tracks_by_ids(track_ids):
   tracks = []
   for track_id in track_ids:
      tracks.append((Track.query.filter(Track.track_id == track_id).first()).to_dict())
   return tracks
   

def get_tracks_without_mentioned_by_ids(track_ids):
    tracks = Track.query.filter(~Track.track_id.in_(track_ids)).all()
    return [track.to_dict() for track in tracks]

   


######################################################################################################
def prepare_features(tracks_data):
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
   
def encode_artist_ids(tracks_data):
   unique_artist_ids = {track['artist_id'] for track in tracks_data}
   
   artist_id_to_int = {artist_id: idx for idx, artist_id in enumerate(unique_artist_ids)}
   
   for track in tracks_data:
      track['artist_id'] = artist_id_to_int[track['artist_id']]
   
   return tracks_data

def cluster_tracks(tracks_data):
   tracks_data = encode_artist_ids(tracks_data)

   features = prepare_features(tracks_data)

   kmeans = KMeans(n_clusters=TASTE_GROUPS)
   clusters = kmeans.fit_predict(features)


   result = [[] for _ in range(TASTE_GROUPS)]  

   for i, track in enumerate(tracks_data):
      result[clusters[i]].append(track)

   return result


######################################################################################################


def prepare_features_without_discrete(tracks_data):
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


def recommend_tracks_for_cluster(tracks_data):
   liked_tracks_ids = [track["track_id"] for track in tracks_data]
   propositions_pool = get_tracks_without_mentioned_by_ids(liked_tracks_ids)
   
   liked_tracks_features = prepare_features_without_discrete(tracks_data)
   propositions_features = prepare_features_without_discrete(propositions_pool)

   average_features = np.mean(liked_tracks_features, axis=0)

   similarities = cosine_similarity(propositions_features, [average_features])
   
   propositions_with_similarity = [(track, similarity[0]) for track, similarity in zip(propositions_pool, similarities)]
   propositions_with_similarity.sort(key=lambda x: x[1], reverse=True)
   recommended_tracks = [track["track_id"] for track, _ in propositions_with_similarity[:CLUSTER_RECOMMENDATION]]
   
   return recommended_tracks
######################################################################################################

def evaluate_tracks(train_data, test_data):
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
######################################################################    

def test_tree_accuracy():
   users_id = [101,
    202, 303, 404, 505, 606, 707, 808, 909,
    102, 202, 302, 402, 502, 602, 702, 802, 902
   ]

   pred = []
   base = []
   delt = []

   results  = ""

   for user_id in users_id:

      train_data = get_weighed_tracks(user_id)


      train_data = dict(list(train_data.items())[20:])
      test_data = dict(list(train_data.items())[:20])


      tracks_data = [track_id for track_id in test_data.keys()]

      prediction = evaluate_tracks(train_data, test_data)

      for track in tracks_data:
         results += f" {track}  : {round(prediction[track], 3)} : {round(test_data.get(track, 0), 3)} ---- {round(abs(prediction[track] - test_data.get(track, 0)), 3)}\n"
         pred.append(prediction[track])
         base.append(test_data.get(track, 0))
         delt.append(abs(prediction[track] - test_data.get(track, 0)))
   results += f"\n\n\-------\n\n min :  {round(np.min(delt), 3)}\n max : {round(np.max(delt), 3)}\n ddelt ---- {round(np.mean(delt), 3)}\n\n\-------\n\n"
   return str(results)

######################################################################################################



def recommend_for_group(user_ids):
   
   tracks_ids = get_top_tracks(user_ids)
   tracks_data = get_tracks_by_ids(tracks_ids)
   track_clusters = cluster_tracks(tracks_data)

   recommendations = []

   for cluster in track_clusters:
      recommendations += recommend_tracks_for_cluster(cluster)

   #tree time  - recommendations, user_ids
   
   data = {}
   for user_id in user_ids:
      predictions = evaluate_tracks(get_weighed_tracks(user_id), recommendations)
      data = {key: data.get(key, 0) + predictions.get(key, 0) for key in data | predictions}

   data = sorted(data, key=data.get, reverse=True)[:FINAL_PLAYLIST_LENGTH]


   return data

