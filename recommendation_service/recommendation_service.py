
from active_gen import UpdateGroupReccomendations
from flask import Flask, request
from init_gen import GroupReccomendations

from analize import (check_clusters, check_create_recommendations, check_features,
                   check_recommendation, check_tree_accuracy)

app = Flask(__name__)



@app.route("/check_recommendations", methods=["POST"])
def check_recommendations():
    user_ids = request.get_json()
    model = GroupReccomendations(user_ids)
    response = check_create_recommendations(model)
    return response


@app.route("/check_clusters", methods=["POST"])
def check_clustering():
    user_ids = request.get_json()
    model = GroupReccomendations(user_ids)
    response = check_clusters(model)
    return response


@app.route("/check_tree", methods=["POST"])
def check_tree():
    user_ids = request.get_json()
    model = GroupReccomendations(user_ids)
    response = check_tree_accuracy(model)
    return response


@app.route("/check_update", methods=["POST"])
def check_update_accuracy():
    user_ids = request.get_json()
    model = UpdateGroupReccomendations("mock_playlist_id")
    response = check_recommendation(model, user_ids)
    return response


@app.route("/check_features", methods=["POST"])
def check_features_used():
    user_ids = request.get_json()
    model = GroupReccomendations(user_ids)
    response = check_features(model)
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
