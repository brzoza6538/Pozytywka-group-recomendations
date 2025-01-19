from flask import Flask, request
from init_gen import GroupReccomendations
from active_gen import UpdateGroupReccomendations
from tests import test_create_recommendations, test_features, test_clusters, test_recommendation
import requests
from datetime import datetime

app = Flask(__name__)
app_url = "http://app:8000"



def get_type_of_tracks(user_id, event_type, from_time, to_time=datetime.utcnow()):
    data = [user_id, event_type, from_time.isoformat(), to_time.isoformat()]
    user_records = (requests.post(f"{app_url}/users_actions_of_type", json=data)).json()
    return user_records

def get_tracks_by_ids(track_ids):
    tracks = (requests.post(f"{app_url}/track_by_id", json=track_ids)).json()
    return tracks


def get_tracks_without_mentioned_by_ids(track_ids):
    tracks = (
        requests.post(f"{app_url}/tracks_without_mentioned_by", json=track_ids)
    ).json()
    return tracks

def get_tracks_and_reactions_for_playlist(playlist_id):
    tracks = (requests.post(f"{app_url}/tracks_of_playlist", json=playlist_id)).json()
    return tracks


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



@app.route("/test_recommendations", methods=["POST"])
def check_recommendations():
    user_ids = request.get_json()
    model = GroupReccomendations(user_ids)
    response = test_create_recommendations(model)
    return response

@app.route("/test_clusters", methods=["POST"])
def test_clustering():
    user_ids = request.get_json()
    model = GroupReccomendations(user_ids)
    response = test_clusters(test_clusters())
    return response

@app.route("/test_tree", methods=["POST"])
def test_tree_accuracy():
    user_ids = request.get_json()
    model = GroupReccomendations(user_ids)
    response = test_tree_accuracy(model)
    return response


@app.route("/test_update", methods=["POST"])
def test_update_accuracy():
    user_ids = request.get_json()
    model = UpdateGroupReccomendations("mock_playlist_id")
    response = test_recommendation(model)
    return response

@app.route("/test_features", methods=["POST"])
def test_features_used():
    user_ids = request.get_json()
    model = GroupReccomendations(user_ids)
    response = test_features(model)
    return response



@app.route("/recommend", methods=["POST"])
def recommend():
    user_ids = request.get_json()
    track_ids = GroupReccomendations(user_ids).get_advanced()
    return track_ids


@app.route("/worse_recommend", methods=["POST"])
def worse_recommend():
    user_ids = request.get_json()
    track_ids = GroupReccomendations(user_ids).get_basic()
    return track_ids


@app.route("/update_recommendation", methods=["POST"])
def update_recommendation():
    playlist_id = request.get_json()
    track_ids = UpdateGroupReccomendations(playlist_id).get()
    return track_ids


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8001)
