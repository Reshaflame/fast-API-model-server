def predict_batch(req):
    predictions = []
    toggle = False
    for row in req.data:
        predictions.append({
            "row_id": row.row_id,
            "anomaly": toggle
        })
        toggle = not toggle  # Flip between True/False
    return predictions
