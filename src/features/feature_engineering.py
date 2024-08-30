

def feature_engineering(data):
    data["monthly_hrs_per_proj"] = data["average_montly_hours"] / data["number_project"]
    data["satis_eval_interaction"] = data["satisfaction_level"] * data["last_evaluation"]
    return data