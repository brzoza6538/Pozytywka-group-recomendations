import uuid

import requests
from flask import Blueprint, request

from models import Recommendation, db

recommendation_url = "http://recommendation:8001/"

recommendations_blueprint = Blueprint("recommendation", __name__)


@recommendations_blueprint.route("/recommend", methods=["POST"])
def create_recommendation():
    users_ids = request.get_json()

    track_ids = (
        requests.post(f"{recommendation_url}/recommend", json=users_ids)
    ).json()

    playlist_id = str(uuid.uuid4())[:22]
    for track in track_ids:
        new_recommendation = Recommendation(
            playlist_id=playlist_id, track_id=track)
        db.session.add(new_recommendation)
    db.session.commit()

    return str(playlist_id), 201


@recommendations_blueprint.route("/adapt", methods=["PATCH"])
def update_recommendations():
    data = request.get_json()
    playlist_id = data.get("playlist_id")

    for line in data['selectedRecommendations']:
        recommendation_id = line.get("recommendation_id")
        recommendation = Recommendation.query.get(recommendation_id)

        recommendation.reaction = bool(line.get("checked"))
    db.session.commit()

    track_ids = (
        requests.post(
            f"{recommendation_url}/update_recommendation", json=playlist_id)
    ).json()

    for track in track_ids:
        existing_recommendation = Recommendation.query.filter_by(
            playlist_id=playlist_id, track_id=track
        ).first()
        if not existing_recommendation:
            new_recommendation = Recommendation(
                playlist_id=playlist_id, track_id=track)
            db.session.add(new_recommendation)

    db.session.commit()

    return str(playlist_id), 201
