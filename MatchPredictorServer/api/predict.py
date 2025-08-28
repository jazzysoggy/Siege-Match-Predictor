"""REST API for posts."""
from flask import Flask, request, jsonify
import MatchPredictorServer
from MatchPredictorServer.lib.utilize import predictTest


@MatchPredictorServer.app.route("/api/v1/", methods=["GET"])
def return_services():
    """Return Avaliable Services."""
    service = {
        "predict": "/api/v1/predict/",
    }

    return Flask.jsonify(service), 200


@MatchPredictorServer.app.route("/api/v1/predict/", methods=['POST'])
def predict():
    """Predict."""
    data = request.get_json()
    
    team1 = data.get("team1", [])
    team2 = data.get("team2", [])
    
    results = predictTest(" ".join(team1 + team2),0)
    
    return jsonify({"results": [True] + list(results)})
    