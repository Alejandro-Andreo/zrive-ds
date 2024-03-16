from sklearn.ensemble import HistGradientBoostingClassifier


class Push_Model:
    FEATURE_COLS = [
        "ordered_before",
        "global_popularity",
        "abandoned_before",
        "std_days_to_buy_product_type",
        "days_since_purchase_product_type",
        "avg_days_to_buy_product_type",
        "user_order_seq",
        "avg_days_to_buy_variant_id",
        "set_as_regular",
    ]
    TARGET_COL = "outcome"

    def __init__(self, params: dict):
        self.model = HistGradientBoostingClassifier(**params)
        self.best_params = params

    def fit(self, df):
        self.model.fit(df[self.FEATURE_COLS], df[self.TARGET_COL])

    def predict(self, df):
        return self.model.predict(df[self.FEATURE_COLS])

    def predict_proba(self, df):
        return self.model.predict_proba(df[self.FEATURE_COLS])[:, 1]
