from flask import Flask
import os
import json
from models import db, User, Artist, Session, Track, Recommendation
from data_routes import data_blueprint
from model.model_routes import model_blueprint

def clear_database():
    db.session.query(User).delete()
    db.session.query(Artist).delete()
    db.session.query(Session).delete()
    db.session.query(Track).delete()

    db.session.commit()

def load_data_from_jsonl(file_path, dataType):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    data = json.loads(line)
                    user = dataType(data)
                    db.session.add(user)
                except json.JSONDecodeError as e:
                    print(f"Błąd wczytywania linii JSON: {e}")
        db.session.commit()
    else:
        raise Exception(f"{file_path} : file not found")

def create_app():
    app = Flask(__name__)
    app.config.from_object("config.Config")

    db.init_app(app)

    with app.app_context():
        app.register_blueprint(data_blueprint)
        app.register_blueprint(model_blueprint)

        db.create_all()
        # db.session.query(Recommendation).delete()

        if Artist.query.count() == 0:
            load_data_from_jsonl("data/artists.jsonl", Artist)
        if Session.query.count() == 0:
            load_data_from_jsonl("data/sessions.jsonl", Session)
        if Track.query.count() == 0:
            load_data_from_jsonl("data/tracks.jsonl", Track)
        if User.query.count() == 0:            
            load_data_from_jsonl("data/users.jsonl", User)

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000)
