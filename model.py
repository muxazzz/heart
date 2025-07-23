# Импортируй свою модель или замени заглушку
from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from catboost import Pool

def preprocess_data(df):
    """
    Функция для базовой предобработки данных
    
    Параметры:
    df (pandas.DataFrame): Входной датафрейм
    
    Возвращает:
    pandas.DataFrame: Предобработанный датафрейм
    """
    # Создаем копию датафрейма
    df_processed = df.copy()
    
    # 1. Заменим названия столбцов на snake_case
    for col in df_processed.columns:
        df_processed.rename(columns={col: col.strip().replace(' ',"_")}, inplace=True)
    
    # 2. Обработка пропущенных значений
    # Удаление строк, где все значения отсутствуют
    #df_processed.dropna(how='all', inplace=True)
    
    # 3. Приведение названий колонок к нижнему регистру
    df_processed.columns = df_processed.columns.str.lower()
    
    # 4. Изучаем количественные переменные
    numeric_columns = df_processed.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_columns:
        count_NA_num = df_processed[col].isna().sum()
        print(f"Количество пропусков в колонке {col}: {count_NA_num} \n")
        
    # 5. Изучаем категориальные значения
    categorical_columns = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        count_NA_cat = df_processed[col].isna().sum()
        print(f"Количество пропусков в колонке {col}: {count_NA_cat}")
        print(f"Уникальные значения {col}: {df_processed[col].unique()} \n")
    
    # 6. Удаление дубликатов
    df_processed.drop_duplicates(inplace=True)
    
    
    # 7. Удаление пробелов в строковых значениях
    for col in categorical_columns:
        df_processed[col] = df_processed[col].str.strip()
    
    return df_processed


# class HeartRiskModel:
#     def __init__(self):
#         self.model = self._train_model()
    
#     def preprocess_input(self, features: list) -> pd.DataFrame:
#         columns = [
#             'age',
#             'cholesterol',
#             'heart_rate',
#             'family_history',
#             'smoking',
#             'obesity',
#             'alcohol_consumption',
#             'exercise_hours_per_week',
#             'diet',
#             'medication_use',
#             'stress_level',
#             'sedentary_hours_per_day',
#             'income',
#             'bmi',
#             'triglycerides',
#             'physical_activity_days_per_week',
#             'sleep_hours_per_day',
#             'blood_sugar',
#             'troponin',
#             'systolic_blood_pressure',
#             'diastolic_blood_pressure'
#         ]
#         df = pd.DataFrame(features, columns=columns)
#         if 'heart_attack_risk_(binary)' in df.columns:
#             df = df.drop(columns=['heart_attack_risk_(binary)'])
#         return df

#     def _train_model(self):
#         df = pd.read_csv("heart_train.csv", index_col=0)
#         df = preprocess_data(df)
#         df = df.dropna()
#         df = df.loc[df['troponin'] < 0.08]
#         df = df.loc[df['ck-mb'] < 0.1]
#         df = df.loc[df['blood_sugar'] < 0.4]
        
        
#         X = df.drop(columns=['heart_attack_risk_(binary)','id', 'ck-mb', 'previous_heart_problems', 'gender', 'diabetes'], axis=1)
#         y = df['heart_attack_risk_(binary)']
#         model = CatBoostClassifier(
#             random_state=42,
#             verbose=100,
#             iterations=10,
#             depth=16,
#             learning_rate=1,
#             l2_leaf_reg=1,
#             #border_count = 32,
#             loss_function='Logloss',
#             eval_metric='Recall'
#         )
#         model.fit(X, y)
#         return model
    

#     def predict(self, features: list) -> float:
#         X = self.preprocess_input(features)
#         proba = self.model.predict_proba(X)[0][1]
#         prediction = 'Да' if proba >= 0.2 else 'Нет'
#         return round(proba * 100, 2)


class HeartRiskModel:
    def __init__(self):
        self.model = CatBoostClassifier()
        self.model.load_model("heart.cbm")  # заранее обученная модель
        self.features = [
            'age',
            'cholesterol',
            'heart_rate',
            'exercise_hours_per_week',
            'sedentary_hours_per_day',
            'income',
            'bmi',
            'triglycerides',
            'sleep_hours_per_day',
            'gender',
            'systolic_blood_pressure',
            'phys', 'f1', 
            'pressure'
        ]
        self.cat_features = ['gender']
    def preprocess_input(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = preprocess_data(df)
        df = df.dropna()
        df['phys'] = df['physical_activity_days_per_week'] / 7.0
        df['f1'] = df['sedentary_hours_per_day']/df['exercise_hours_per_week']
        df['pressure'] = df['systolic_blood_pressure']/df['diastolic_blood_pressure']
        df = df[['age',
            'cholesterol',
            'heart_rate',
            'sedentary_hours_per_day',
            'income', 'bmi',
            'triglycerides', 'sleep_hours_per_day',
            'gender', 'systolic_blood_pressure',
            'phys', 'f1', 
            'pressure']]
        if "heart_attack_risk_(binary)" in df.columns:
            df.drop(columns=["heart_attack_risk_(binary)"], inplace=True)
        return df

    def predict(self, df: pd.DataFrame, threshold: float = 0.1):
        df = self.preprocess_input(df)
        probabilities = self.model.predict_proba(df)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        return probabilities, predictions, df.index
    
    
heart_model = HeartRiskModel()