
from active_gen import UpdateGroupReccomendations
from flask import Flask, request
from init_gen import GroupReccomendations
from tests import (test_clusters, test_create_recommendations, test_features,
                   test_recommendation, test_tree_accuracy)

app = Flask(__name__)



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
    response = test_clusters(model)
    return response


@app.route("/test_tree", methods=["POST"])
def test_tree():
    user_ids = request.get_json()
    model = GroupReccomendations(user_ids)
    response = test_tree_accuracy(model)
    return response


@app.route("/test_update", methods=["POST"])
def test_update_accuracy():
    user_ids = request.get_json()
    model = UpdateGroupReccomendations("mock_playlist_id")
    response = test_recommendation(model, user_ids)
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
