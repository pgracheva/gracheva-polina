"""
Основной файл с решением соревнования
Здесь должен быть весь ваш код для создания предсказаний
"""
def create_submission(predictions):
    """
    Пропишите здесь создание файла submission.csv в папку results
    !!! ВНИМАНИЕ !!! ФАЙЛ должен иметь именно такого названия
    """

    # Создать пандас таблицу submission

    import os
    import pandas as pd
    os.makedirs('results', exist_ok=True)
    submission_path = 'results/submission.csv'
    
    submission = predictions
    submission.to_csv(submission_path, index=False)
    
    print(f"Submission файл сохранен: {submission_path}")
    
    return submission_path


def main():
    """
    Главная функция программы
    
    Вы можете изменять эту функцию под свои нужды,
    но обязательно вызовите create_submission() в конце!
    """
    print("=" * 50)
    print("Запуск решения соревнования")
    print("=" * 50)

    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    from catboost import CatBoostRegressor
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.decomposition import PCA
    from scipy.optimize import minimize
    import warnings
    warnings.filterwarnings('ignore')
    
    # 1. Класс для детекции аномалий
    class TemporalPCABasedAnomalyDetector:
        def __init__(self, window_size=7, n_components=3):
            self.window_size = window_size
            self.n_components = n_components
            self.pca_models = {}
            self.feature_cols = None
            self.fitted_ = False
            self.global_mean_score = 0.0
    
        def _build_windows(self, X):
            windows = []
            for i in range(len(X)):
                start = max(0, i - self.window_size + 1)
                window = X.iloc[start:i+1].values
                if len(window) < self.window_size:
                    pad = np.repeat(window[:1], self.window_size - len(window), axis=0)
                    window = np.vstack([pad, window])
                windows.append(window.flatten())
            return np.array(windows)
    
        def detect_anomalies(self, X, product_ids):
            self.feature_cols = X.columns.tolist()
            scores = np.zeros(len(X))
            for pid in product_ids.unique():
                idx = product_ids == pid
                X_pid = X.loc[idx]
                if len(X_pid) < self.window_size + 1:
                    continue
                windows = self._build_windows(X_pid)
                pca = PCA(
                    n_components=min(self.n_components, windows.shape[1]),
                    random_state=322
                )
                recon = pca.inverse_transform(pca.fit_transform(windows))
                error = ((windows - recon) ** 2).mean(axis=1)
                scores[idx.values] = error
                self.pca_models[pid] = pca
            self.anomaly_scores_ = scores
            self.global_mean_score = np.mean(scores[scores > 0]) if np.any(scores > 0) else 0.0
            self.threshold_ = np.percentile(scores, 95)
            self.fitted_ = True
            anomalies = np.where(scores > self.threshold_, -1, 1)
            return anomalies, scores
    
        def get_anomaly_features(self, X, product_ids):
            if not self.fitted_:
                raise ValueError("Detector not fitted")
            scores = np.full(len(X), self.global_mean_score)
            for pid, pca in self.pca_models.items():
                idx = product_ids == pid
                X_pid = X.loc[idx]
                if len(X_pid) < self.window_size:
                    continue
                windows = self._build_windows(X_pid)
                recon = pca.inverse_transform(pca.transform(windows))
                error = ((windows - recon) ** 2).mean(axis=1)
                scores[idx.values] = error
            return pd.DataFrame({
                'ts_recon_error': scores,
                'ts_anomaly_flag': (scores > self.threshold_).astype(int)
            }, index=X.index)
    
    # 2. Класс для снижения размерности с LDA
    class LDADimensionalityReducer:
        def __init__(self, n_components=None, random_state=322):
            self.requested_n_components = n_components
            self.random_state = random_state
            self.scaler = StandardScaler()
            self.lda = None
            self.is_fitted = False
            self.n_components_ = 0
            self.feature_names_in_ = None
    
        def fit_transform(self, X, y_labels):
            X = X.copy()
            self.feature_names_in_ = list(X.columns)
            X_scaled = self.scaler.fit_transform(X)
            n_features = X_scaled.shape[1]
            n_classes = len(np.unique(y_labels))
            # Максимально допустимое число компонент для LDA
            max_components = min(n_features, max(0, n_classes - 1))
            if max_components <= 0:
                self.is_fitted = False
                self.n_components_ = 0
                return pd.DataFrame(index=X.index)
            # Определяем реально используемое число компонент
            if self.requested_n_components is None:
                n_comp = max_components
            else:
                n_comp = int(min(self.requested_n_components, max_components))
            self.n_components_ = n_comp
    
            self.lda = LinearDiscriminantAnalysis(n_components=n_comp, solver='svd')
            X_reduced = self.lda.fit_transform(X_scaled, y_labels)
            self.is_fitted = True
            if X_reduced.ndim == 1:
                X_reduced = X_reduced.reshape(-1, 1)
            columns = [f'lda_component_{i+1}' for i in range(X_reduced.shape[1])]
            result = pd.DataFrame(X_reduced, columns=columns, index=X.index)
            print(f" Количество классов: {n_classes}")
            print(f" Снижение размерности: {n_features} -> {result.shape[1]} признаков")
            return result
    
        def transform(self, X):
            X = X.copy()
            if not self.is_fitted:
                return pd.DataFrame(index=X.index)
           
            missing_cols = set(self.feature_names_in_) - set(X.columns)
            if missing_cols:
                X = X.copy()
                for col in missing_cols:
                    X[col] = 0
           
            X_scaled = self.scaler.transform(X[self.feature_names_in_])
            X_reduced = self.lda.transform(X_scaled)
           
            if X_reduced.ndim == 1:
                X_reduced = X_reduced.reshape(-1, 1)
               
            columns = [f'lda_component_{i+1}' for i in range(X_reduced.shape[1])]
            return pd.DataFrame(X_reduced, columns=columns, index=X.index)
    
    # 3. Класс для квантильной регрессии 
    class QuantileRegressorWithIoU:
        def __init__(self, lower_quantile=0.05, upper_quantile=0.95, epsilon=1e-8):
            self.lower_quantile = lower_quantile
            self.upper_quantile = upper_quantile
            self.epsilon = epsilon
            self.model_lower = None
            self.model_upper = None
            self.model_lower_cb = None
            self.model_upper_cb = None
            self.feature_columns = None
            self.lower_shift = 0.0
            self.upper_shift = 0.0
    
        def calculate_iou(self, y_true_lower, y_true_upper, pred_lower, pred_upper):
            width_true = y_true_upper - y_true_lower + self.epsilon
            width_pred = pred_upper - pred_lower + self.epsilon
            intersection = np.maximum(
                0,
                np.minimum(y_true_upper, pred_upper) - np.maximum(y_true_lower, pred_lower)
            )
            union = width_true + width_pred - intersection
            iou = intersection / (union + 1e-12)
            return iou
    
        def train(self, X_train, y_lower_train, y_upper_train,
                  X_val=None, y_lower_val=None, y_upper_val=None):
            self.feature_columns = X_train.columns.tolist()
            print(f" Количество признаков: {len(self.feature_columns)}")
    
            params_lower = {
                'objective': 'quantile',
                'alpha': self.lower_quantile,
                'learning_rate': 0.02,
                'num_leaves': 50,
                'max_depth': 7,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'min_child_samples': 20,
                'random_state': 322,
                'n_jobs': -1,
                'verbose': -1,
                'n_estimators': 1000
            }
            params_upper = params_lower.copy()
            params_upper['alpha'] = self.upper_quantile
    
            self.model_lower = lgb.LGBMRegressor(**params_lower)
            self.model_upper = lgb.LGBMRegressor(**params_upper)
    
            # CatBoost
            params_lower_cb = {
                'iterations': 1000,
                'depth': 7,
                'learning_rate': 0.02,
                'random_seed': 322,
                'verbose': 0,
                'thread_count': -1,
                'loss_function': f'Quantile:alpha={self.lower_quantile}'
            }
            params_upper_cb = params_lower_cb.copy()
            params_upper_cb['loss_function'] = f'Quantile:alpha={self.upper_quantile}'
    
            self.model_lower_cb = CatBoostRegressor(**params_lower_cb)
            self.model_upper_cb = CatBoostRegressor(**params_upper_cb)
    
            if X_val is not None and y_lower_val is not None and y_upper_val is not None:
               
                callbacks = [
                    lgb.callback.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.callback.log_evaluation(period=0)
                ]
               
                self.model_lower.fit(
                    X_train, y_lower_train,
                    eval_set=[(X_val, y_lower_val)],
                    eval_metric='quantile',
                    callbacks=callbacks
                )
               
                self.model_upper.fit(
                    X_train, y_upper_train,
                    eval_set=[(X_val, y_upper_val)],
                    eval_metric='quantile',
                    callbacks=callbacks
                )
    
                self.model_lower_cb.fit(
                    X_train, y_lower_train,
                    eval_set=(X_val, y_lower_val),
                    early_stopping_rounds=50
                )
    
                self.model_upper_cb.fit(
                    X_train, y_upper_train,
                    eval_set=(X_val, y_upper_val),
                    early_stopping_rounds=50
                )
               
                # Оптимизация сдвигов
                self._optimize_shifts(X_val, y_lower_val, y_upper_val)
            else:
                self.model_lower.fit(X_train, y_lower_train)
                self.model_upper.fit(X_train, y_upper_train)
                self.model_lower_cb.fit(X_train, y_lower_train)
                self.model_upper_cb.fit(X_train, y_upper_train)
    
        def _optimize_shifts(self, X_val, y_lower_val, y_upper_val):
            #Оптимизация сдвигов границ для максимизации IoU
       
            pred_lower_lgb = self.model_lower.predict(X_val)
            pred_upper_lgb = self.model_upper.predict(X_val)
            pred_lower_cb = self.model_lower_cb.predict(X_val)
            pred_upper_cb = self.model_upper_cb.predict(X_val)
            pred_lower = (pred_lower_lgb + pred_lower_cb) / 2
            pred_upper = (pred_upper_lgb + pred_upper_cb) / 2
            def objective(shifts):
                lower_shift, upper_shift = shifts
                adjusted_lower = pred_lower + lower_shift
                adjusted_upper = pred_upper + upper_shift
                mask = adjusted_lower > adjusted_upper
                adjusted_upper[mask] = adjusted_lower[mask] + self.epsilon
                iou = self.calculate_iou(y_lower_val.values, y_upper_val.values, adjusted_lower, adjusted_upper)
                return -iou.mean()
            initial_guess = [0.0, 0.0]
            bounds = [(-0.5, 0.5), (-0.5, 0.5)]
            try:
                result = minimize(objective, initial_guess, bounds=bounds,
                                  method='L-BFGS-B', options={'maxiter': 200})
                self.lower_shift, self.upper_shift = result.x
                print(f" Оптимальные сдвиги: lower={self.lower_shift:.4f}, upper={self.upper_shift:.4f}")
                print(f" IoU после калибровки: {-result.fun:.4f}")
            except Exception as e:
                print(" Оптимизация не удалась:", e)
                self.lower_shift, self.upper_shift = 0.0, 0.0
                
        def predict(self, X):
            X_aligned = X[self.feature_columns] if self.feature_columns is not None else X
            pred_lower_lgb = self.model_lower.predict(X_aligned)
            pred_upper_lgb = self.model_upper.predict(X_aligned)
            pred_lower_cb = self.model_lower_cb.predict(X_aligned)
            pred_upper_cb = self.model_upper_cb.predict(X_aligned)
            pred_lower = (pred_lower_lgb + pred_lower_cb) / 2 + self.lower_shift
            pred_upper = (pred_upper_lgb + pred_upper_cb) / 2 + self.upper_shift
            mask = pred_lower > pred_upper
            pred_upper[mask] = pred_lower[mask] + self.epsilon
            pred_lower = np.clip(pred_lower, 0, None)
            pred_upper = np.clip(pred_upper, pred_lower + self.epsilon, None)
            return pred_lower, pred_upper
        
    # 4. Утилиты для признаков 
    def create_basic_features(df, encoders=None, fit_encoders=False):
        df_processed = df.copy()
        encoders_local = {} if encoders is None else encoders.copy()
        # Сортировка по дате и продукту для лагов
        if 'dt' in df_processed.columns and 'product_id' in df_processed.columns:
            df_processed['dt'] = pd.to_datetime(df_processed['dt'])
            df_processed = df_processed.sort_values(['product_id', 'dt'])
        # Базовые временные фичи 
        if 'dt' in df_processed.columns:
            df_processed['year'] = df_processed['dt'].dt.year
            df_processed['month'] = df_processed['dt'].dt.month
            df_processed['day'] = df_processed['dt'].dt.day
            df_processed['day_of_year'] = df_processed['dt'].dt.dayofyear
            df_processed['week_of_year'] = df_processed['dt'].dt.isocalendar().week # Добавляем неделю
        if 'dow' in df_processed.columns:
            df_processed['dow_sin'] = np.sin(2 * np.pi * df_processed['dow'] / 7)
            df_processed['dow_cos'] = np.cos(2 * np.pi * df_processed['dow'] / 7)
        if 'month' in df_processed.columns:
            df_processed['month_sin'] = np.sin(2 * np.pi * df_processed['month'] / 12)
            df_processed['month_cos'] = np.cos(2 * np.pi * df_processed['month'] / 12)
        # Взаимодействия 
        if all(col in df_processed.columns for col in ['n_stores', 'holiday_flag']):
            df_processed['stores_holiday'] = df_processed['n_stores'] * df_processed['holiday_flag']
        if all(col in df_processed.columns for col in ['n_stores', 'activity_flag']):
            df_processed['stores_activity'] = df_processed['n_stores'] * df_processed['activity_flag']
        # Квадраты и корни для числовых
        numeric_cols = ['n_stores', 'precpt', 'avg_temperature', 'avg_humidity', 'avg_wind_level']
        for col in numeric_cols:
            if col in df_processed.columns:
                df_processed[f'{col}_squared'] = df_processed[col] ** 2
                df_processed[f'{col}_sqrt'] = np.sqrt(np.maximum(df_processed[col], 0)) # Добавляем sqrt для нелинейности
        # Лаги и rolling по продуктам 
        if 'product_id' in df_processed.columns and 'dt' in df_processed.columns:
            for col in numeric_cols + ['price_p05', 'price_p95'] if 'price_p05' in df_processed.columns else numeric_cols: # Лаги таргетов только в train
                for lag in [1, 3, 7]: # Лаги 1,3,7 дней
                    df_processed[f'{col}_lag_{lag}'] = df_processed.groupby('product_id')[col].shift(lag)
                # Rolling mean/std (окно 3-7 дней)
                df_processed[f'{col}_rolling_mean_3'] = df_processed.groupby('product_id')[col].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
                df_processed[f'{col}_rolling_std_7'] = df_processed.groupby('product_id')[col].rolling(window=7, min_periods=1).std().reset_index(0, drop=True)
            # Заполняем NaN в лагах 
            for col in df_processed.columns:
                if '_lag_' in col or '_rolling_' in col:
                    df_processed[col] = df_processed.groupby('product_id')[col].transform(lambda x: x.fillna(x.mean()))
        
        cat_cols = ['product_id', 'management_group_id', 'first_category_id', 'second_category_id', 'third_category_id']
        for col in cat_cols:
            if col in df_processed.columns:
                if fit_encoders and 'price_p05' in df_processed.columns: # Target encoding в train
                    mean_enc = df_processed.groupby(col)['price_p05'].mean() 
                    df_processed[f'{col}_mean_enc'] = df_processed[col].map(mean_enc)
                    encoders_local[f'{col}_mean_enc'] = mean_enc
                elif not fit_encoders and f'{col}_mean_enc' in encoders_local:
                    df_processed[f'{col}_mean_enc'] = df_processed[col].map(encoders_local[f'{col}_mean_enc']).fillna(0)
              
                if fit_encoders:
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                    encoders_local[col] = le
                else:
                    if col in encoders_local:
                        le = encoders_local[col]
                        vals = df_processed[col].astype(str).tolist()
                        transformed = []
                        classes = set(le.classes_.astype(str))
                        for v in vals:
                            if v in classes:
                                transformed.append(int(le.transform([v])[0]))
                            else:
                                transformed.append(-1)
                        df_processed[col] = transformed
        if 'dt' in df_processed.columns:
            df_processed = df_processed.drop(columns=['dt'])
        return df_processed, encoders_local
    def create_labels_for_lda(y_lower, n_classes=10, method='quantile'):
        if method == 'quantile':
            labels = pd.qcut(y_lower, q=n_classes, labels=False, duplicates='drop')
        elif method == 'equal_width':
            labels = pd.cut(y_lower, bins=n_classes, labels=False)
        elif method == 'kmeans':
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_classes, random_state=322)
            labels = kmeans.fit_predict(y_lower.values.reshape(-1, 1))
        print(f" Количество классов: {len(np.unique(labels))}")
        return labels
    
    # 5. Финальный pipeline для обучения и предсказания 
    def train_final_and_predict(train_df, test_df, use_lda=True, n_lda_components=10, contamination=0.05, calib_fraction=0.1):
        if 'dt' in train_df.columns:
            train_df = train_df.sort_values('dt').reset_index(drop=True)
     
        X_train_raw, encoders = create_basic_features(train_df.drop(['price_p05', 'price_p95'], axis=1), encoders=None, fit_encoders=True)
        y_train_lower = train_df['price_p05'].reset_index(drop=True)
        y_train_upper = train_df['price_p95'].reset_index(drop=True)
    
        X_test_raw, _ = create_basic_features(test_df.copy(), encoders=encoders, fit_encoders=False)
       
        # Детекция аномалий
        numeric_cols_train = X_train_raw.select_dtypes(include=[np.number]).columns.tolist()
        anomaly_detector = TemporalPCABasedAnomalyDetector(
            window_size=7,
            n_components=3
        )
        anomalies, scores = anomaly_detector.detect_anomalies(
            X_train_raw[numeric_cols_train],
            train_df['product_id']
        )
        anomaly_features_train = anomaly_detector.get_anomaly_features(
            X_train_raw[numeric_cols_train],
            train_df['product_id']
        )
        anomaly_features_test = anomaly_detector.get_anomaly_features(
            X_test_raw[numeric_cols_train],
            test_df['product_id']
        )
        X_train_with_anomaly = pd.concat([X_train_raw.reset_index(drop=True), anomaly_features_train.reset_index(drop=True)], axis=1)
        X_test_with_anomaly = pd.concat([X_test_raw.reset_index(drop=True), anomaly_features_test.reset_index(drop=True)], axis=1)
        
        # LDA 
        if use_lda:
    
            y_train_labels = create_labels_for_lda(y_train_lower, n_classes=10, method='quantile')
            numeric_cols_lda = X_train_with_anomaly.select_dtypes(include=[np.number]).columns.tolist()
           
            lda_reducer = LDADimensionalityReducer(n_components=n_lda_components, random_state=322)
            lda_features_train = lda_reducer.fit_transform(
                X_train_with_anomaly[numeric_cols_lda],
                y_train_labels
            )
            # Может вернуться пустая таблица
            if lda_features_train.shape[1] == 0:
                X_train_final = X_train_with_anomaly.copy()
                lda_features_test = lda_reducer.transform(X_test_with_anomaly[numeric_cols_lda])
                X_test_final = X_test_with_anomaly.copy()
            else:
                lda_features_test = lda_reducer.transform(X_test_with_anomaly[numeric_cols_lda])
                X_train_final = pd.concat([X_train_with_anomaly.reset_index(drop=True), lda_features_train.reset_index(drop=True)], axis=1)
                X_test_final = pd.concat([X_test_with_anomaly.reset_index(drop=True), lda_features_test.reset_index(drop=True)], axis=1)
        else:
            X_train_final = X_train_with_anomaly.copy()
            X_test_final = X_test_with_anomaly.copy()
            
        # Обучение модели
        n_samples = len(X_train_final)
        min_calib_rows = 100
        calibrate = (calib_fraction > 0) and (int(n_samples * calib_fraction) >= min_calib_rows)
        if calibrate:
            split_idx = int(n_samples * (1 - calib_fraction))
            X_fit = X_train_final.iloc[:split_idx].reset_index(drop=True)
            y_fit_lower = y_train_lower.iloc[:split_idx].reset_index(drop=True)
            y_fit_upper = y_train_upper.iloc[:split_idx].reset_index(drop=True)
            X_calib = X_train_final.iloc[split_idx:].reset_index(drop=True)
            y_calib_lower = y_train_lower.iloc[split_idx:].reset_index(drop=True)
            y_calib_upper = y_train_upper.iloc[split_idx:].reset_index(drop=True)
            print(f"Калибровка сдвигов: обучаем на {len(X_fit)} строках, калибруем на {len(X_calib)} строках")
        else:
            X_fit = X_train_final.reset_index(drop=True)
            y_fit_lower = y_train_lower.reset_index(drop=True)
            y_fit_upper = y_train_upper.reset_index(drop=True)
            X_calib = None
            y_calib_lower = None
            y_calib_upper = None
            print("Калибровка сдвигов пропущена")
        quantile_regressor = QuantileRegressorWithIoU(lower_quantile=0.05, upper_quantile=0.95, epsilon=1e-8)
        if X_calib is not None:
            quantile_regressor.train(X_fit, y_fit_lower, y_fit_upper, X_calib, y_calib_lower, y_calib_upper)
        else:
            quantile_regressor.train(X_fit, y_fit_lower, y_fit_upper)
            
        # Предсказание на тесте
        # Убедимся, что колонки теста содержат те же колонки
        for c in quantile_regressor.feature_columns:
            if c not in X_test_final.columns:
                X_test_final[c] = 0.0
       
        X_test_final = X_test_final[quantile_regressor.feature_columns]
        test_pred_lower, test_pred_upper = quantile_regressor.predict(X_test_final)
    
        submission = pd.DataFrame({
        'row_id': range(len(test_pred_lower)),
        'price_p05': test_pred_lower,
        'price_p95': test_pred_upper
        })
        submission = submission[['row_id', 'price_p05', 'price_p95']]
    
        submission.to_csv('results/submission.csv', index=False)
        return submission, quantile_regressor
    
#    if __name__ == "__main__":
    train_path = "data/train.csv"
    test_path = "data/test.csv"
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    submission_df, model = train_final_and_predict(train_df, test_df, use_lda=True, n_lda_components=10)
    print(submission_df.head())
    
##    create_submission(submission)
    
    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)


if __name__ == "__main__":
    main()
