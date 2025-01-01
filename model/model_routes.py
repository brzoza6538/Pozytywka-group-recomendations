from flask import Blueprint, jsonify, request, render_template
from models import Recommendation, Artist, User, Track, Session, db

model_blueprint = Blueprint("recommendation", __name__)


def get_recommendations_by_playlist_id(playlist_id):
    reccommendations = Recommendation.query.filter(Recommendation.playlist_id == playlist_id).all()
    return reccommendations

@model_blueprint.route('/')
def index():
    return render_template('index.html')

@model_blueprint.route("/recommend/<playlist_id>", methods=["GET"])
def get_recommendation(playlist_id):
    recommendations = (
        Recommendation.query
        .filter(Recommendation.playlist_id == playlist_id)
        .order_by(Recommendation.id.desc()) 
        .limit(5)
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
