from flask import Blueprint, jsonify, request, render_template
from models import Recommendation, Artist, User, Track, Session, db
import random
from model.init_gen import recommend_for_group, test_tree_accuracy
from model.active_gen import recommend_more
import uuid


model_blueprint = Blueprint("recommendation", __name__)


@model_blueprint.route('/')
def index():
    return render_template('index.html')

@model_blueprint.route("/recommended_playlist/<playlist_id>", methods=["GET"])
def get_recommended_playlist(playlist_id):
    recommendations = (
        db.session.query(Recommendation, Track, Artist)
        .join(Track, Track.track_id == Recommendation.track_id) 
        .join(Artist, Artist.id == Track.artist_id) 
        .filter(Recommendation.playlist_id == playlist_id, Recommendation.reaction == True)
        .order_by(Recommendation.id.desc())
        .all()
    )

    return jsonify([
        {
            "reaction": recommendation.reaction,
            "id": recommendation.id,
            "playlist_id": recommendation.playlist_id,
            "track_name": track.name,
            "artist_name": artist.name,
            "track_id": track.track_id
        }
        for recommendation, track, artist in recommendations
    ])


@model_blueprint.route("/recommend/<playlist_id>", methods=["GET"])
def get_recommendation(playlist_id):
    recommendations = (
        db.session.query(Recommendation, Track, Artist)
        .join(Track, Track.track_id == Recommendation.track_id) 
        .join(Artist, Artist.id == Track.artist_id) 
        .filter(Recommendation.playlist_id == playlist_id, Recommendation.reaction == None)
        .order_by(Recommendation.id.desc())
        .all()
    )

    return jsonify([
        {
            "reaction": recommendation.reaction,
            "id": recommendation.id,
            "playlist_id": recommendation.playlist_id,
            "track_name": track.name,
            "artist_name": artist.name,
            "track_id": track.track_id
        }
        for recommendation, track, artist in recommendations
    ])


@model_blueprint.route("/recommend", methods=["POST"])
def create_recommendation():
    users_ids = []

    users_ids = request.get_json()

    track_ids = recommend_for_group(users_ids)

    playlist_id = str(uuid.uuid4())[:22]
    for track in track_ids:
        new_recommendation = Recommendation(playlist_id=playlist_id, track_id=track)
        db.session.add(new_recommendation)
    db.session.commit()

    return str(playlist_id), 201


###############################################################################

@model_blueprint.route("/adapt", methods=["PATCH"])
def update_recommendations():
    data = request.get_json()
    playlist_id = data.get("playlist_id")

    for line in data['selectedRecommendations']:
        recommendation_id = line.get("recommendation_id")
        recommendation = Recommendation.query.get(recommendation_id)

        recommendation.reaction = bool(line.get("checked"))
    db.session.commit()



    track_ids = recommend_more(playlist_id)

    for track in track_ids:
        existing_recommendation = Recommendation.query.filter_by(playlist_id=playlist_id, track_id=track).first()
        if not existing_recommendation:
            new_recommendation = Recommendation(playlist_id=playlist_id, track_id=track)
            db.session.add(new_recommendation)

    db.session.commit()

    return str(playlist_id), 201



##############################################################################




@model_blueprint.route("/check", methods=["POST"])
def mock_test():
    data = request.get_json()
    return test_tree_accuracy()


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
    return jsonify([artist.to_dict() for artist in artists]), 200
