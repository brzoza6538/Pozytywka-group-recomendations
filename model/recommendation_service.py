from flask import Flask, request
from init_gen import GroupReccomendations
from active_gen import UpdateGroupReccomendations


app = Flask(__name__)


@app.route("/test_recommendations", methods=["POST"])
def check_recommendations():
    users_ids = request.get_json()
    response = GroupReccomendations(users_ids).test_create_recommendations()
    return response

@app.route("/test_clusters", methods=["POST"])
def test_clustering():
    users_ids = request.get_json()
    response = GroupReccomendations(users_ids).test_clusters()
    return response

@app.route("/test_tree", methods=["POST"])
def test_tree_accuracy():
    users_ids = request.get_json()
    response = GroupReccomendations(users_ids).test_tree_accuracy()
    return response

@app.route("/recommend", methods=["POST"])
def recommend():
    users_ids = request.get_json()
    track_ids = GroupReccomendations(users_ids).get()
    return track_ids


@app.route("/update_recommendation", methods=["POST"])
def update_recommendation():
    playlist_id = request.get_json()
    track_ids = UpdateGroupReccomendations(playlist_id).get()
    return track_ids


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8001)
