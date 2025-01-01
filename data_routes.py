from flask import Blueprint, jsonify, request
from models import Artist, User, Track, Session, db

data_blueprint = Blueprint("api", __name__)

@data_blueprint.route("/users", methods=["GET"])
def get_users():
    users = User.query.all()
    return jsonify([user.to_dict() for user in users])


# @data_blueprint.route("/users", methods=["POST"])
# def create_user():
#     data = request.get_json()
#     new_user = User(data)
#     db.session.add(new_user)
#     db.session.commit()
#     return jsonify(new_user.to_dict()), 201

#------------------------

@data_blueprint.route("/artists", methods=["GET"])
def get_artists():
    artists = Artist.query.all()
    return jsonify([artist.to_dict() for artist in artists])

# @data_blueprint.route("/artists", methods=["POST"])
# def create_artist():
#     data = request.get_json()
#     new_artist = Artist(data)
#     db.session.add(new_artist)
#     db.session.commit()
#     return jsonify(new_artist.to_dict()), 201

#------------------------


@data_blueprint.route("/sessions", methods=["GET"])
def get_sessions():
    sessions = Session.query.all()
    return jsonify([session.to_dict() for session in sessions])

# @data_blueprint.route("/sessions", methods=["POST"])
# def create_session():
#     data = request.get_json()
#     new_session = Session(data)
#     db.session.add(new_session)
#     db.session.commit()
#     return jsonify(new_session.to_dict()), 201


#------------------------


@data_blueprint.route("/tracks", methods=["GET"])
def get_tracks():
    tracks = Track.query.all()
    return jsonify([track.to_dict() for track in tracks])

# @data_blueprint.route("/tracks", methods=["POST"])
# def create_track():
#     data = request.get_json()
#     new_track = Track(data)
#     db.track.add(new_track)
#     db.track.commit()
#     return jsonify(new_track.to_dict()), 201