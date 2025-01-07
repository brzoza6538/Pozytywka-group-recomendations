from flask import Blueprint, jsonify, request, render_template
from models import Recommendation, Artist, User, Track, Session, db
from datetime import datetime, timedelta

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter


TIME_BORDER = datetime.utcnow() - timedelta(days=90)
USERS_LIKED_TRACKS_AMOUNT = 100
CLUSTER_RECOMMENDATION = 10
TASTE_GROUPS = 10 #TODO zależne od ilości osób? 

TRAIN_DATA_AMOUNT = 0

LIKE_WEIGHT = 5
SKIP_WEIGHT = -5
PLAY_WEIGHT = 4

FINAL_PLAYLIST_LENGTH = 10

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
   

# def get_tracks_by_ids(track_ids): # added artist data for checking results
#     tracks_with_artists = (
#         db.session.query(Track, Artist)
#         .join(Artist, Track.artist_id == Artist.id)
#         .filter(Track.track_id.in_(track_ids))
#         .all()
#     )

#     tracks = []
#     for track, artist in tracks_with_artists:
#         track_dict = track.to_dict()
#         artist_dict = artist.to_dict()

#         track_dict['artist'] = artist_dict
#         tracks.append(track_dict)

#     return tracks

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



def evaluate_tracks(user_id, tracks_id):
    """
    Przyjmuje ID użytkownika, sesję użytkownika oraz dane utworów,
    a następnie zwraca oceny piosenek na podstawie modelu drzewa decyzyjnego.

    :param user_id: ID użytkownika
    :param tracks_id: Lista ID utworów do oceny
    :return: Słownik {track_id: ocena}
    """
    # Przygotowanie danych treningowych
    training_data = get_weighed_tracks(user_id)

    # Przygotowanie cech i etykiet dla danych treningowych
    tracks_data = [get_tracks_by_ids([track_id])[0] for track_id in training_data.keys()]
    features_list = prepare_features_without_discrete(tracks_data)
    labels = list(training_data.values())  # Oceny dla treningowych utworów

    # Budowa modelu drzewa decyzyjnego
    tree = DecisionTreeRegressor(max_depth=5)  # Maksymalna głębokość drzewa (można dostosować)
    tree.fit(features_list, labels)  # Trening drzewa na danych

    # Pobranie utworów, które chcemy ocenić
    tracks = get_tracks_by_ids(tracks_id)
    predictions = {}

    # Przewidywanie ocen dla każdego utworu
    for track in tracks:
        try:
            # Przygotowanie cech dla konkretnego utworu
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
            # Obliczanie score'u (wyniku) utworu za pomocą drzewa
            score = tree.predict([track_features])[0]
            # Normalizacja wyniku na przedział 0-1 (opcjonalnie)
            predictions[track["track_id"]] = max(0, min(1, score))
        except KeyError as e:
            raise ValueError(f"Brak wymaganej cechy w danych utworów: {e}")

    return predictions


def test_tree_accuracy():
   users_id = [
    101, 202, 303, 404, 505, 606, 707, 808, 909,
    102, 202, 302, 402, 502, 602, 702, 802, 902
   ]

   # pred = []
   # base = []
   delt = []

   for user_id in users_id:
      training_data = get_weighed_tracks(user_id)

      prediction = evaluate_tracks(user_id, training_data)
      results  = ""

      for track in training_data.keys():
         #results += f" {track}  : {round(prediction[track], 3)} : {round(training_data[track], 3)} ---- {round(abs(prediction[track] - training_data[track]), 3)}\n"
         # pred.append(prediction[track])
         # base.append(training_data[track])
         delt.append(abs(prediction[track] - training_data[track]))

   return round(np.mean(delt), 3)

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
      x = evaluate_tracks(user_id, recommendations)
      data = {key: data.get(key, 0) + x.get(key, 0) for key in data | x}

   data = sorted(data, key=data.get, reverse=True)[:FINAL_PLAYLIST_LENGTH]


   return data



# example of PATHC data 

# [
#     {
#         "recommendation_id": "1",
#         "checked": true
#     },
#     {
#         "recommendation_id": "2",
#         "checked": false
#     },
#     {
#         "recommendation_id": "3",
#         "checked": true
#     }
# ]