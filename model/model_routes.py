from flask import Blueprint, jsonify, request, render_template
from models import Recommendation, Artist, User, Track, Session, db
import random
from model.recc_gen_module import recommend_for_group
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





import logging
from flask import jsonify

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG)

@model_blueprint.route("/adapt", methods=["PATCH"])
def update_recommendations():
    data = request.get_json()
    playlist_id = data.get("playlist_id")
    
    # Diagnostyka - sprawdzamy dane wejściowe
    if not playlist_id:
        logging.error("Playlist ID is missing.")
        return jsonify({"error": "Playlist ID is required"}), 400
    
    if 'selectedRecommendations' not in data:
        logging.error("Selected recommendations are missing.")
        return jsonify({"error": "Selected recommendations are required"}), 400
    
    selected_recommendations = data.get('selectedRecommendations', [])
    
    if not selected_recommendations:
        logging.warning("No recommendations selected.")
    
    # Aktualizacja reakcji dla każdej rekomendacji
    for line in selected_recommendations:
        recommendation_id = line.get("recommendation_id")
        checked = bool(line.get("checked"))
        
        if not recommendation_id or checked is None:
            logging.error(f"Invalid data for recommendation: {line}")
            continue  # Pomija nieprawidłowe dane, ale kontynuuje przetwarzanie
        
        recommendation = Recommendation.query.get(recommendation_id)
        if recommendation:
            recommendation.reaction = checked
            logging.info(f"Updated recommendation {recommendation_id} with reaction {checked}")
        else:
            logging.warning(f"Recommendation with ID {recommendation_id} not found.")
    
    try:
        db.session.commit()
        logging.info("Changes committed successfully.")
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error during commit: {e}")
        return jsonify({"error": "Failed to update recommendations"}), 500

    return str(playlist_id), 201

    



@model_blueprint.route("/check", methods=["POST"])
def mock_test():
    data = request.get_json()
    return recommend_for_group(data), 201



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
