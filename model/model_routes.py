from flask import Blueprint, jsonify, request
from models import Recomendation, Artist, User, Track, Session, db

model_blueprint = Blueprint("reccomend", __name__)


def get_recomendations_by_playlist_id(playlist_id):
    reccomendations = Recomendation.query.filter(Recomendation.playlist_id == playlist_id).all()
    return reccomendations

@model_blueprint.route("/reccomend", methods=["POST"])
def create_recomendation():
    users_id = []

    data = request.get_json()
    for line in data:
        users_id.append(line["user_id"])



#MOCK
    # for i in range(10):
    #     for j in range(i):
    #         new_recomendation = Recomendation(playlist_id=i, track_id=j)
    #         db.session.add(new_recomendation)
    #         db.session.commit()

    check = get_recomendations_by_playlist_id(7)[-5:] # szukamy ostatnich pięciu z playlisty dla zaawansowanej budowy

    return jsonify([x.to_dict() for x in check]), 201
