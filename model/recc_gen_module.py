from flask import Blueprint, jsonify, request, render_template
from models import Recommendation, Artist, User, Track, Session, db
from datetime import datetime, timedelta

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


TIME_BORDER = datetime.utcnow() - timedelta(days=90)
USERS_LIKED_TRACKS_AMOUNT = 100
CLUSTER_RECOMMENDATION = 10
TASTE_GROUPS = 10 #TODO zależne od ilości osób? 
#znajdujesz z tego piosenki które wszystkim się podobają więc może być ich od cholery
#dopiero później skróć do n najlepiej ocenianych gdzie n jest na tyle małe że generowanie na bieżąco biędzie miało czas zadzialać

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



def get_played_tracks(user_id):
   started_tracks = get_started_tracks(user_id)
   skipped_tracks = get_skipped_tracks(user_id)
   tracks_counter = 0

   for track_id in list(started_tracks.keys()):
      if(track_id in skipped_tracks.keys()):
         started_tracks[track_id] -= skipped_tracks[track_id]

      tracks_counter += started_tracks[track_id]

      if started_tracks[track_id] <= 0:
         del started_tracks[track_id]


   for track_id in started_tracks.keys():
      started_tracks[track_id] = started_tracks[track_id] / tracks_counter

   return started_tracks


def get_top_tracks(user_ids):
   connected_scores = {} 

   for user_id in user_ids:
      user_scores = get_played_tracks(user_id)

      for track_id in user_scores.keys():
         if(track_id not in connected_scores.keys()):
            connected_scores[track_id] = user_scores[track_id]
         else:
            connected_scores[track_id] += user_scores[track_id]

   sorted_items = sorted(connected_scores.items(), key=lambda item: item[1], reverse=True)
   top_tracks = [track_id for track_id, score in sorted_items[:USERS_LIKED_TRACKS_AMOUNT]]
   return top_tracks


# def get_tracks_by_ids(track_ids):
#    tracks = []
#    for track_id in track_ids:
#       tracks.append((Track.query.filter(Track.track_id == track_id).first()).to_dict())
#    return tracks
   

def get_tracks_by_ids(track_ids): # added artist data for checking results
    tracks_with_artists = (
        db.session.query(Track, Artist)
        .join(Artist, Track.artist_id == Artist.id)
        .filter(Track.track_id.in_(track_ids))
        .all()
    )

    tracks = []
    for track, artist in tracks_with_artists:
        track_dict = track.to_dict()
        artist_dict = artist.to_dict()

        track_dict['artist'] = artist_dict
        tracks.append(track_dict)

    return tracks

def get_tracks_without_mentioned_by_ids(track_ids):
    tracks = Track.query.filter(~Track.track_id.in_(track_ids)).all()
    return [track.to_dict() for track in tracks]

   


######################################################################################################
def prepare_feature(tracks_data):
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

def prepare_features_without_disrete(tracks_data):
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


def recommend_tracks(tracks_data):
   liked_tracks_ids = [track["track_id"] for track in tracks_data]
   propositions_pool = get_tracks_without_mentioned_by_ids(liked_tracks_ids)
   
   liked_tracks_features = prepare_features_without_disrete(tracks_data)
   propositions_features = prepare_features_without_disrete(propositions_pool)

   average_features = np.mean(liked_tracks_features, axis=0)

   similarities = cosine_similarity(propositions_features, [average_features])
   
   propositions_with_similarity = [(track, similarity[0]) for track, similarity in zip(propositions_pool, similarities)]
   propositions_with_similarity.sort(key=lambda x: x[1], reverse=True)
   recommended_tracks = [track for track, _ in propositions_with_similarity[:CLUSTER_RECOMMENDATION]]
   
   return recommended_tracks
######################################################################################################
def module_1(data):
   
   tracks_ids = get_top_tracks(data)
   tracks_data = get_tracks_by_ids(tracks_ids)
   track_clusters = cluster_tracks(tracks_data)

   # release = ""

   # for cluster in track_clusters:
   #    release += f"{[ track['name'] for track in cluster]} \n\n----------\n\n"

   recommendations = []
   release = ""

   for cluster in track_clusters:
      recommendations += recommend_tracks(cluster)

   ids = [track['track_id'] for track in recommendations]

   release += f"{get_tracks_by_ids(ids)} \n\n----------\n\n" 

   

   return release
   


   #  recommendations = (
   #      Recommendation.query
   #      .filter(Recommendation.playlist_id == 45)
   #      .order_by(Recommendation.id.desc()) 
   #      .limit(3)
   #      .all()
   #  )

   #  return jsonify([recommendation.to_dict() for recommendation in recommendations])

   # sample_data = [
   #    {"id": "6kD1SNGPkfX9LwaGd1FG92", "name": "Put Your Dreams Away (For Another Day)", "popularity": 53, "duration_ms": 186173, "explicit": 0, "artist_id": "1Mxqyy3pSjf8kZZL4QVxS0", "release_date": "1944", "danceability": 0.197, "energy": 0.0546, "key": 1, "loudness": -22.411, "speechiness": 0.0346, "acousticness": 0.95, "instrumentalness": 0.276, "liveness": 0.152, "valence": 0.1, "tempo": 90.15},
   #    {"id": "4Pnzw1nLOpDNV6MKI5ueIR", "name": "Nancy (With the Laughing Face) - 78rpm Version", "popularity": 55, "duration_ms": 199000, "explicit": 0, "artist_id": "1Mxqyy3pSjf8kZZL4QVxS0", "release_date": "1944", "danceability": 0.295, "energy": 0.0826, "key": 1, "loudness": -19.569, "speechiness": 0.0367, "acousticness": 0.984, "instrumentalness": 0.000358, "liveness": 0.156, "valence": 0.169, "tempo": 128.6},
   #    {"id": "7GLmfKOe5BfOXk7334DoKt", "name": "Saturday Night (Is The Loneliest Night In The Week)", "popularity": 54, "duration_ms": 163000, "explicit": 0, "artist_id": "1Mxqyy3pSjf8kZZL4QVxS0", "release_date": "1944", "danceability": 0.561, "energy": 0.335, "key": 9, "loudness": -11.093, "speechiness": 0.0499, "acousticness": 0.84, "instrumentalness": 1.52e-06, "liveness": 0.788, "valence": 0.59, "tempo": 126.974},
   #    {"id": "6JpN5w95em8SODPiM7W2PH", "name": "The Story of O.J.", "popularity": 66, "duration_ms": 231760, "explicit": 1, "artist_id": "3nFkdlSjzX9mRTtwJOzDYB", "release_date": "2017-07-07", "danceability": 0.741, "energy": 0.718, "key": 7, "loudness": -5.823, "speechiness": 0.415, "acousticness": 0.283, "instrumentalness": 2.21e-06, "liveness": 0.23, "valence": 0.576, "tempo": 165.848},
   #    {"id": "1gT5TGwbkkkUliNzHRIGi1", "name": "4:44", "popularity": 65, "duration_ms": 284493, "explicit": 1, "artist_id": "3nFkdlSjzX9mRTtwJOzDYB", "release_date": "2017-07-07", "danceability": 0.261, "energy": 0.852, "key": 9, "loudness": -4.965, "speechiness": 0.158, "acousticness": 0.139, "instrumentalness": 4.26e-05, "liveness": 0.477, "valence": 0.431, "tempo": 177.997},
   #    {"id": "3nDYsXggRQxf7PCNUjR4rz", "name": "Dead Presidents II", "popularity": 57, "duration_ms": 266067, "explicit": 1, "artist_id": "3nFkdlSjzX9mRTtwJOzDYB", "release_date": "1996-06-25", "danceability": 0.758, "energy": 0.912, "key": 0, "loudness": -8.758, "speechiness": 0.349, "acousticness": 0.172, "instrumentalness": 1.09e-06, "liveness": 0.149, "valence": 0.573, "tempo": 87.335},
   #    {"id": "0ZHu7jkSSrT0eK4OxuG4O5", "name": "Excuse Me Miss", "popularity": 58, "duration_ms": 281240, "explicit": 1, "artist_id": "3nFkdlSjzX9mRTtwJOzDYB", "release_date": "2002-11-12", "danceability": 0.714, "energy": 0.862, "key": 6, "loudness": -5.531, "speechiness": 0.286, "acousticness": 0.0305, "instrumentalness": 0.0, "liveness": 0.0884, "valence": 0.887, "tempo": 92.849},

   # ]