from model.module2_prediction import PredictionModel

predict = PredictionModel()

test_1 = [
  {
    "session_id": 124,
    "timestamp": "2023-10-25T10:15:32.065944",
    "user_id": 551,
    "track_id": "0RNxWy0PC3AyH4ThH3aGK6",
    "event_type": "advertisment"
  },
  {
    "session_id": 124,
    "timestamp": "2023-10-25T10:15:32.065944",
    "user_id": 551,
    "track_id": "0RNxWy0PC3AyH4ThH3aGK6",
    "event_type": "play"
  },
  {
    "session_id": 124,
    "timestamp": "2023-10-25T10:16:32.065944",
    "user_id": 551,
    "track_id": "0RNxWy0PC3AyH4ThH3aGK6",
    "event_type": "play"
  },
  {
    "session_id": 124,
    "timestamp": "2023-10-25T10:17:32.065944",
    "user_id": 551,
    "track_id": "0RNxWy0PC3AyH4ThH3aGK6",
    "event_type": "play"
  }
]

track = {
  "id": "0RNxWy0PC3AyH4ThH3aGK6",
  "name": "Mack the Knife",
  "popularity": 55,
  "duration_ms": 201467,
  "explicit": 0,
  "id_artist": "19eLuQmk9aCobbVDHc6eek",
  "release_date": "1929",
  "danceability": 0.673,
  "energy": 0.377,
  "key": 0,
  "loudness": -14.141,
  "speechiness": 0.0697,
  "acousticness": 0.586,
  "instrumentalness": 0.0,
  "liveness": 0.332,
  "valence": 0.713,
  "tempo": 88.973
}

data = predict.prepare_all_data([551], [test_1], ["0RNxWy0PC3AyH4ThH3aGK6"])
predict.train(data[0], data[1])
print(predict.predict("0RNxWy0PC3AyH4ThH3aGK6", track))