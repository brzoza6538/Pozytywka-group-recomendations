from recommendation_service import get_tracks_by_ids, get_tracks_without_mentioned_by_ids, get_type_of_tracks
from active_gen import enumerate_artist_id

from datetime import datetime, timedelta
import itertools

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np

def test_create_recommendations(rec_class):
    """
    test different parameters and methods of GroupReccomendations

    """
    model_p = [rec_class.create_recommendations_advanced, rec_class.create_recommendations_basic]
    time_start_p = [360]
    time_end_p = [180]
    users_favourite_tracks_amount_p = [50 * len(rec_class.user_ids)]
    cluster_recommendation_p = [10]
    taste_groups_p = [5 * len(rec_class.user_ids)]

    liked_weight_p = [5]
    skipped_weight_p = [-3]
    started_weight_p = [3]

    normalisation_range_up_p = [1]
    normalisation_range_down_p = [-1]

    rec_class._final_playlist_length = 100

    constraints = list(
        itertools.product(
            model_p,
            time_start_p,
            time_end_p,
            users_favourite_tracks_amount_p,
            cluster_recommendation_p,
            taste_groups_p,

            liked_weight_p,
            skipped_weight_p,
            started_weight_p,
            normalisation_range_up_p,
            normalisation_range_down_p
        )
    )
    message = "(timeframe_start, timeframe_end, users_favourite_tracks_amount, cluster_recommendation, taste_groups, liked_weight, skipped_weight, started_weight, score_normalisation_upper_limit, score_normalisation_lower_limit)"
    results = [message]
    print(message)

    for setup in constraints:
        rec_class._time_window_start = datetime.utcnow() - timedelta(days=setup[1])
        rec_class._time_window_end = datetime.utcnow() - timedelta(days=setup[2])

        rec_class._users_favourite_tracks_amount = setup[3]
        rec_class._cluster_recommendation = setup[4]
        rec_class._taste_groups = setup[5]

        rec_class._liked_weight = setup[6]
        rec_class._skipped_weight = setup[7]
        rec_class._started_weight = setup[8]

        rec_class.normalisation_range_up = setup[9]
        rec_class.normalisation_range_down = setup[10]

        start = datetime.utcnow()
        recommendations = setup[0]()
        end = datetime.utcnow()

        rec_class._time_window_start = datetime.utcnow() - timedelta(days=setup[2])
        rec_class._time_window_end = datetime.utcnow()
        test_tracks = rec_class.get_top_tracks()

        duplicates = 0
        for track in recommendations:
            if track in test_tracks:
                duplicates += 1
        message = (
            f"""
            \n{setup[0].__name__}
            {setup[1:]}
            - Generated: {len(recommendations)}
            - Test sample size: {len(test_tracks)}
            - Found duplicates: {duplicates}
            - Score (duplicates/test tracks): {round(duplicates / len(recommendations), 4)}
            - Time taken: {(end - start).total_seconds()}
            """
        )
        results.append(message)
        print(message)

    return results






def test_features(rec_class):
    """
        test feature combinations
    """
    model_p = [rec_class.create_recommendations_advanced, rec_class.create_recommendations_basic]
    rec_class._final_playlist_length = 100
    
    time_constraint_up = 360
    time_constraint_down = 180

    features_chosen = ["valence", "acousticness"]
    filtered_features = [f for f in rec_class._used_features if (f not in features_chosen)]

    results = []

    rec_class._final_playlist_length = 100

    for feature_tested in ([filtered_features] + list(itertools.combinations(filtered_features, len(filtered_features)-1))):
        rec_class._used_features = feature_tested

        for model in model_p:
            rec_class._time_window_start = datetime.utcnow() - timedelta(days=time_constraint_up)
            rec_class._time_window_end = datetime.utcnow() - timedelta(days=time_constraint_down)

            start = datetime.utcnow()
            recommendations = model()
            end = datetime.utcnow()

            rec_class._time_window_start = datetime.utcnow() - timedelta(days=time_constraint_down)
            rec_class._time_window_end = datetime.utcnow()
            test_tracks = rec_class.get_top_tracks()

            duplicates = 0
            for track in recommendations:
                if track in test_tracks:
                    duplicates += 1
            message = (
                f"""
                \n{model.__name__}
                {rec_class._used_features}
                - Generated: {len(recommendations)}
                - Test sample size: {len(test_tracks)}
                - Found duplicates: {duplicates}
                - Score (duplicates/test tracks): {round(duplicates / len(recommendations), 4)}
                - Time taken: {(end - start).total_seconds()}
                """
            )
            results.append(message)
            print(message)

    return results




def test_clusters(rec_class): 
    # to chyba nic nie mówi - to wynika z braku danych przy za dużej ilośći klastrów niż czegokolwiek innego
    tracks_ids = rec_class.get_top_tracks()
    tracks_data = get_tracks_by_ids(tracks_ids)

    accuracy = []
    for i in [2, 8, 32, 48, 128]:
        track_clusters = rec_class.cluster_tracks(tracks_data, i)

        X = []
        y = []

        for cluster_idx, cluster in enumerate(track_clusters):
            for track in cluster:
                features = rec_class.prepare_features([track], discrete=False)[0]
                X.append(features)
                y.append(cluster_idx)

        X = np.array(X)
        y = np.array(y)

        model = RandomForestClassifier(n_estimators=100)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))

        print(f"{i} -- {accuracy}")
    return str(accuracy)


def test_recommendation(update_class, user_ids):
    """
        test algorithm training it on users's history of last year, test based on last 3 months 
    """
    time_window_start = datetime.utcnow() - timedelta(days=360)
    time_window_end = datetime.utcnow()- timedelta(days=90)

    results = []
    accuracy_counter = 0 
    d_counter = 0 

    for user_id in user_ids:
        data = update_class.get_user_data(user_id, time_window_start, time_window_end) [:30]

        track_ids = {track['track_id'] for track in data}
        track_ids = list(track_ids)
        
        propositions =  update_class.get_user_data(user_id, time_window_end, datetime.utcnow())
        propositions = [{key: value for key, value in track.items() if key != "reaction"} for track in propositions]

        propositions, data = enumerate_artist_id(propositions, data)
        predictions = update_class.predict(data, propositions)

        test_data = update_class.get_user_data(user_id, time_window_end, datetime.utcnow())
        accuracy_counter = 0 
        d_counter = 0 

        for i in range(len(predictions)):
            if (predictions[i] == 1 and test_data[i]['reaction'] == True):
                accuracy_counter += 1
            else:
                d_counter += 1
        print(round(accuracy_counter/len(test_data), 4))
        results.append(f"{user_id}   {round(accuracy_counter/len(test_data), 4)}")
    return results

def test_tree_accuracy(update_class):
    '''
    used to test accuracy of decision tree based scores

    setup = (liked_weight_p, skipped_weight_p, started_weight_p, depths, normalisation_range_up_p, normalisation_range_down_p ) 
    '''
    liked_weight_p = [1, 5]
    skipped_weight_p = [-5, -10]
    started_weight_p = [1, 3]
    depths = [10]
    normalisation_range_up_p = [1]
    normalisation_range_down_p = [0]
      
    constraints = list(
        itertools.product(
            liked_weight_p,
            skipped_weight_p,
            started_weight_p,
            depths,
            normalisation_range_up_p,
            normalisation_range_down_p
        )
    )
    results = []

    for setup in constraints:
        update_class._liked_weight = setup[0]
        update_class._skipped_weight = setup[1]
        update_class._started_weight = setup[2]

        update_class.normalisation_range_up = setup[4]
        update_class.normalisation_range_down = setup[5]
        diff = []
        for user_id in update_class.user_ids:
            train_data = update_class.get_weighed_tracks(user_id)

            train_data = list(train_data.items())

            train_items, test_items = train_test_split(train_data, test_size=0.2, random_state=42)

            train_data = dict(train_items)
            test_data = dict(test_items)

            tracks_data = [track_id for track_id in test_data.keys()]

            prediction = update_class.evaluate_tracks(train_data, test_data, setup[3])

            for track in tracks_data:
                diff.append(abs(prediction[track] - test_data.get(track, 0)))
        results.append(f"min :  {round(np.min(diff), 3)} max : {round(np.max(diff), 3)} ddelt : {round(np.mean(diff), 3)} \nsetup : {setup}")
    return results


