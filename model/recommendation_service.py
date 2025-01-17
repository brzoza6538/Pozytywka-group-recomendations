from flask import Flask, request
from init_gen import GroupReccomendations
from active_gen import UpdateGroupReccomendations


# /recommend_tracks = GroupReccomendations(users_ids).get()
# /update_recommendations = UpdateGroupReccomendations(playlist_id).get()
# /test = GroupReccomendations(data).test_create_recommendations()
 


 
app = Flask(__name__)

@app.route("/check", methods=["POST"])
def check():
    data = request.get_json()
    return ("heellerr")

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
