from datetime import datetime

import requests

app_url = "http://app:8000"


def get_type_of_tracks(user_id, event_type, from_time, to_time=datetime.utcnow()):
    data = [user_id, event_type, from_time.isoformat(), to_time.isoformat()]
    user_records = (requests.post(
        f"{app_url}/users_actions_of_type", json=data)).json()
    return user_records


def get_tracks_by_ids(track_ids):
    tracks = (requests.post(f"{app_url}/track_by_id", json=track_ids)).json()
    return tracks


def get_tracks_without_mentioned_by_ids(track_ids):
    tracks = (
        requests.post(f"{app_url}/tracks_without_mentioned_by", json=track_ids)
    ).json()
    return tracks


def get_tracks_and_reactions_for_playlist(playlist_id):
    tracks = (requests.post(
        f"{app_url}/tracks_of_playlist", json=playlist_id)).json()
    return tracks


def get_tracks_by_ids(track_ids):
    tracks = (requests.post(f"{app_url}/track_by_id", json=track_ids)).json()
    return tracks


def get_tracks_without_mentioned_by_ids(track_ids):
    tracks = (
        requests.post(f"{app_url}/tracks_without_mentioned_by", json=track_ids)
    ).json()
    return tracks


def get_type_of_tracks(user_id, event_type, from_time, to_time=datetime.utcnow()):
    data = [user_id, event_type, from_time.isoformat(), to_time.isoformat()]
    user_records = (requests.post(
        f"{app_url}/users_actions_of_type", json=data)).json()
    return user_records