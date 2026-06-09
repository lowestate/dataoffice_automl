from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

try:
    from xgboost import XGBRegressor  # type: ignore
except:
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor  # type: ignore
except:
    LGBMRegressor = None
try:
    from catboost import CatBoostRegressor  # type: ignore
except:
    CatBoostRegressor = None
    
def get_models(random_state: int, plan_id: int = 3):
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "DecisionTree": DecisionTreeRegressor(random_state=random_state),
        "RandomForest": RandomForestRegressor(random_state=random_state),
        "KNN": KNeighborsRegressor(),
        "SVR": SVR(),
    }
    
    if plan_id >= 2:
        models["GradientBoosting"] = GradientBoostingRegressor(random_state=random_state)
        if XGBRegressor:
            models["XGBoost"] = XGBRegressor(random_state=random_state)
        if LGBMRegressor:
            models["LightGBM"] = LGBMRegressor(random_state=random_state)
        if CatBoostRegressor:
            models["CatBoost"] = CatBoostRegressor(verbose=0, random_state=random_state)
    
    return models
