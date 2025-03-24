from datetime import datetime

from flask import Blueprint, jsonify, render_template, request

from models import Artist, Recommendation, Session, Track, User, db

app_blueprint = Blueprint("data", __name__)


@app_blueprint.route("/users", methods=["GET"])
def get_users():
    users = User.query.all()
    return jsonify([user.to_dict() for user in users])


@app_blueprint.route("/track_by_id", methods=["POST"])
def get_tracks_by_ids():
    track_ids = request.get_json()

    tracks = []

    for track_id in track_ids:
        tracks.append(
            (Track.query.filter(Track.track_id == track_id).first()).to_dict()
        )
    return tracks


@app_blueprint.route("/tracks_without_mentioned_by", methods=["POST"])
def get_tracks_without_mentioned_by_ids():
    track_ids = request.get_json()
    tracks = Track.query.filter(~Track.track_id.in_(track_ids)).all()

    return [track.to_dict() for track in tracks]


@app_blueprint.route("/users_actions_of_type", methods=["POST"])
def get_type_of_tracks():
    data = request.get_json()
    user_id, event_type, from_time, to_time = data

    from_time = datetime.fromisoformat(from_time)
    to_time = datetime.fromisoformat(to_time)

    sessions = Session.query.filter(
        (Session.user_id == user_id) &
        (Session.event_type == event_type) &
        (Session.timestamp > from_time) &
        (Session.timestamp <= to_time)
    ).all()

    user_records = {}

    for session in sessions:
        if session.track_id not in user_records.keys():
            user_records[session.track_id] = 0
        user_records[session.track_id] += 1

    return user_records


@app_blueprint.route("/tracks_of_playlist", methods=["POST"])
def get_tracks_and_reactions_for_playlist():
    playlist_id = request.get_json()

    recommendations = (
        db.session.query(Recommendation, Track, Artist)
        .join(Track, Track.track_id == Recommendation.track_id)
        .join(Artist, Artist.id == Track.artist_id)
        .filter(Recommendation.playlist_id == playlist_id)
        .order_by(Recommendation.id.desc())
        .all()
    )

    return [
        {**track.to_dict(), "reaction": int(recommendation.reaction)}
        for recommendation, track, _ in recommendations
    ]


@app_blueprint.route('/')
def index():
    return render_template('index.html')


@app_blueprint.route("/recommended_playlist/<playlist_id>", methods=["GET"])
def get_recommended_playlist(playlist_id):
    recommendations = (
        db.session.query(Recommendation, Track, Artist)
        .join(Track, Track.track_id == Recommendation.track_id)
        .join(Artist, Artist.id == Track.artist_id)
        .filter(
            Recommendation.playlist_id == playlist_id, Recommendation.reaction == True
        )
        .order_by(Recommendation.id.desc())
        .all()
    )

    return jsonify(
        [
            {
                "reaction": recommendation.reaction,
                "id": recommendation.id,
                "playlist_id": recommendation.playlist_id,
                "track_name": track.name,
                "artist_name": artist.name,
                "track_id": track.track_id,
            }
            for recommendation, track, artist in recommendations
        ]
    )


@app_blueprint.route("/recommend/<playlist_id>", methods=["GET"])
def get_recommendation(playlist_id):
    recommendations = (
        db.session.query(Recommendation, Track, Artist)
        .join(Track, Track.track_id == Recommendation.track_id)
        .join(Artist, Artist.id == Track.artist_id)
        .filter(
            Recommendation.playlist_id == playlist_id, Recommendation.reaction == None
        )
        .order_by(Recommendation.id.desc())
        .all()
    )

    return jsonify(
        [
            {
                "reaction": recommendation.reaction,
                "id": recommendation.id,
                "playlist_id": recommendation.playlist_id,
                "track_name": track.name,
                "artist_name": artist.name,
                "track_id": track.track_id,
            }
            for recommendation, track, artist in recommendations
        ]
    )
