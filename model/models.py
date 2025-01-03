from sqlalchemy import Column, Integer, String, ARRAY, Float, Boolean, DateTime
from models import db
from datetime import datetime


class User(db.Model):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    name = Column(String(50))
    city = Column(String(100))
    street = Column(String(200))
    favourite_genres = Column(ARRAY(String))  # Aby przechować listę gatunków
    premium_user = Column(Boolean, nullable=True)  # Nullable dla premium_user

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "city": self.city,
            "street": self.street,
            "favourite_genres": self.favourite_genres,
            "premium_user": self.premium_user
        }


class Artist(db.Model):
    __tablename__ = 'artists'

    id = Column(String(100), primary_key=True)
    name = Column(String(100))
    genres = Column(ARRAY(String))  # Lista gatunków artysty

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "genres": self.genres
        }


class Track(db.Model):
    __tablename__ = 'tracks'

    id = Column(String(100), primary_key=True)
    name = Column(String(100))
    popularity = Column(Integer)
    duration_ms = Column(Integer)
    explicit = Column(Integer)  # 0 lub 1
    # Relacja z artystą
    id_artist = Column(String(100), db.ForeignKey('artists.id'))
    release_date = Column(String(10))
    danceability = Column(Float)
    energy = Column(Float)
    key = Column(Integer)
    loudness = Column(Float)
    speechiness = Column(Float)
    acousticness = Column(Float)
    instrumentalness = Column(Float)
    liveness = Column(Float)
    valence = Column(Float)
    tempo = Column(Float)

    artist = db.relationship("Artist", backref=db.backref('tracks', lazy=True))

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "popularity": self.popularity,
            "duration_ms": self.duration_ms,
            "explicit": self.explicit,
            "id_artist": self.id_artist,
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
            "tempo": self.tempo
        }


class Session(db.Model):
    __tablename__ = 'sessions'

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, unique=True)
    timestamp = Column(DateTime, default=datetime.timezone.utc)
    user_id = Column(Integer, db.ForeignKey('users.id'))
    track_id = Column(String(100), db.ForeignKey('tracks.id'))
    event_type = Column(String(50))  # 'play' lub 'skip'

    user = db.relationship("User", backref=db.backref('sessions', lazy=True))
    track = db.relationship("Track", backref=db.backref('sessions', lazy=True))

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "track_id": self.track_id,
            "event_type": self.event_type
        }
