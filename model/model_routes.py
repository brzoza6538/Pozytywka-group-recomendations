from flask import Blueprint, jsonify, request, render_template
from models import Recommendation, Artist, User, Track, Session, db
import random
from model.recc_gen_module import reccomend_for_group


model_blueprint = Blueprint("recommendation", __name__)


def get_recommendations_by_playlist_id(playlist_id):
    recommendations = Recommendation.query.filter((Recommendation.playlist_id == playlist_id) & (Recommendation.reaction == None)).all()
    return recommendations

@model_blueprint.route('/')
def index():
    return render_template('index.html')

@model_blueprint.route("/recommend/<playlist_id>", methods=["GET"])
def get_recommendation(playlist_id):
    recommendations = (
        Recommendation.query
        .filter(Recommendation.playlist_id == playlist_id)
        .order_by(Recommendation.id.desc()) 
        .limit(3)
        .all()
    )

    return jsonify([recommendation.to_dict() for recommendation in recommendations])


@model_blueprint.route("/recommend", methods=["POST"])
def create_recommendation():
    users_id = []

    data = request.get_json()
    for line in data:
        users_id.append(line["user_id"])



#MOCK 
    # for i in range(10):
    #     for j in range(i):
    #         new_recommendation = Recommendation(playlist_id=i, track_id=j)
    #         db.session.add(new_recommendation)
    #         db.session.commit()


    playlist_id = 7

    return str(playlist_id), 201





@model_blueprint.route("/adapt", methods=["PATCH"])
def update_recomendations():
    reviews = {}

    data = request.get_json()
    for line in data:
        reviews[line["recommendation_id"]] = bool(line["checked"])


    playlist_id = random.randint(6, 8)

    return str(playlist_id), 201

#PATCH nie zmienia kolejności



@model_blueprint.route("/check", methods=["POST"])
def mock_test():
    data = request.get_json()
    return reccomend_for_group(data), 201



# Pobieranie użytkowników
@model_blueprint.route('/users', methods=['GET'])
def get_users():
    users = User.query.all()
    return jsonify([user.to_dict() for user in users]), 200


# Pobieranie sesji
@model_blueprint.route('/sessions', methods=['GET'])
def get_sessions():
    sessions = Session.query.all()
    return jsonify([session.to_dict() for session in sessions]), 200


# Pobieranie utworów
@model_blueprint.route('/tracks', methods=['GET'])
def get_tracks():
    tracks = Track.query.all()
    return jsonify([track.to_dict() for track in tracks]), 200


# Pobieranie artystów
@model_blueprint.route('/artists', methods=['GET'])
def get_artists():
    artists = Artist.query.all()
    return jsonify([artist.to_dict() for ar
tist in artists]), 200
