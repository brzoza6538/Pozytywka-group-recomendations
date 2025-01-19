from datetime import datetime

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class User(db.Model):
    user_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    city = db.Column(db.String(80), nullable=False)
    street = db.Column(db.String(80), nullable=False)
    favourite_genres = db.Column(db.String(255), nullable=False)
    premium_user = db.Column(db.Boolean, nullable=False)

    def __init__(self, data):
        self.user_id = data["user_id"]
        self.name = data["name"]
        self.city = data["city"]
        self.street = data["street"]
        self.premium_user = data["premium_user"]
        self.favourite_genres = ';'.join(data["favourite_genres"])

    def get_favourite_genres(self):
        return self.favourite_genres.split(';') if self.favourite_genres else []

    def to_dict(self):
        return {
            "user_id": self.user_id,
            "name": self.name,
            "city": self.city,
            "street": self.street,
            "favourite_genres": self.get_favourite_genres(),
            "premium_user": self.premium_user,
        }


class Artist(db.Model):
    id = db.Column(db.String(22), primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    genres = db.Column(db.String(255), nullable=False)

    def __init__(self, data):
        self.id = data["id"]
        self.name = data["name"]
        self.genres = ';'.join(data["genres"])

    def get_genres(self):
        return self.genres.split(';') if self.genres else []

    def to_dict(self):
        return {"id": self.id, "name": self.name, "genres": self.get_genres()}


class Track(db.Model):
    track_id = db.Column(db.String(50), primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    popularity = db.Column(db.Integer, nullable=False)  # 0-100
    duration_ms = db.Column(db.Integer, nullable=False)
    explicit = db.Column(db.Boolean, nullable=False)
    artist_id = db.Column(db.String(22), nullable=False)
    release_date = db.Column(db.Integer, nullable=False)  # year only
    danceability = db.Column(db.Float, nullable=False)  # 0.0 - 1.0
    energy = db.Column(db.Float, nullable=False)  # 0.0 - 1.0
    key = db.Column(db.Integer, nullable=False)  # 0-11
    loudness = db.Column(db.Float, nullable=False)  # -60 - 0
    speechiness = db.Column(db.Float, nullable=False)  # 0.0 - 1.0
    acousticness = db.Column(db.Float, nullable=False)  # 0.0 - 1.0
    instrumentalness = db.Column(db.Float, nullable=False)  # 0.0 - 1.0
    liveness = db.Column(db.Float, nullable=False)  # 0.0 - 1.0
    valence = db.Column(db.Float, nullable=False)  # 0.0 - 1.0
    tempo = db.Column(db.Float, nullable=False)

    def __init__(self, data):
        self.track_id = data["id"]
        self.name = data["name"]
        self.popularity = data["popularity"]
        self.duration_ms = data["duration_ms"]
        self.explicit = bool(data["explicit"])
        self.artist_id = data["id_artist"]
        self.release_date = int(data["release_date"][:4])
        self.danceability = data["danceability"]
        self.energy = data["energy"]
        self.key = data["key"]
        self.loudness = data["loudness"]
        self.speechiness = data["speechiness"]
        self.acousticness = data["acousticness"]
        self.instrumentalness = data["instrumentalness"]
        self.liveness = data["liveness"]
        self.valence = data["valence"]
        self.tempo = data["tempo"]

    def to_dict(self):
        return {
            "track_id": self.track_id,
            "name": self.name,
            "popularity": self.popularity,
            "duration_ms": self.duration_ms,
            "explicit": self.explicit,
            "artist_id": self.artist_id,
            "release_date": self.release_date,
            "danceability": self.danceability,
            "energy": self.energy,
            "key": self.key,
            "loudness": self.loudness,
            "speechiness": self.speechiness,
            "acousticness": self.acousticness,
            "instrumentalness": self.instrumentalness,
            "liveness": self.liveness,
            "valence": self.valence,
            "tempo": self.tempo,
        }


class Session(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)  # no nanoseconds
    user_id = db.Column(db.Integer, nullable=False)
    track_id = db.Column(db.String(22), nullable=True)
    event_type = db.Column(
        db.String(20), nullable=False
    )  # skip, play, like, advertisment

    def __init__(self, data):
        self.session_id = data["session_id"]
        self.timestamp = datetime.strptime(
            data["timestamp"][:19], "%Y-%m-%dT%H:%M:%S"
        )  # nanoseconds not included
        self.user_id = data["user_id"]
        self.track_id = data["track_id"]
        self.event_type = data["event_type"]

    def to_dict(self):
        return {
            "action_id": self.id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
            "user_id": self.user_id,
            "track_id": self.track_id,
            "event_type": self.event_type,
        }


class Recommendation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    playlist_id = db.Column(db.String(22), nullable=False)
    track_id = db.Column(db.String(22), nullable=False)
    reaction = db.Column(
        db.Boolean, nullable=True
    )  # skiped, not skipped, not yet decided

    def __init__(self, playlist_id, track_id, reaction=None):
        self.playlist_id = playlist_id
        self.track_id = track_id
        self.reaction = reaction

    def to_dict(self):
        return {
            "recommendation_id": self.id,
            "playlist_id": self.playlist_id,
            "track_id": self.track_id,
            "reaction": self.reaction,
        }
