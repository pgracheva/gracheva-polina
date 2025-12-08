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

    import pandas as pd
    import numpy as np
    import re
    from collections import Counter
    import math
    from catboost import CatBoostRanker, Pool
    from sklearn.model_selection import GroupKFold
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    import warnings
    warnings.filterwarnings('ignore')
    
    RANDOM_SEED = 993
    np.random.seed(RANDOM_SEED)
    
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")
    print()
    
    
    def cleaning_text(text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = ' '.join(text.split())
    
        words = text.split()
        words = [w for w in words if w not in ENGLISH_STOP_WORDS]
        return ' '.join(words)
    
    
    for col in ['query', 'product_title', 'product_description', 'product_bullet_point',
                'product_brand', 'product_color']:
        train_df[f'{col}_clean'] = train_df[col].apply(cleaning_text)
        test_df[f'{col}_clean'] = test_df[col].apply(cleaning_text)
    
    
    train_df['product_text'] = (
        train_df['product_title_clean'] + ' ' +
        train_df['product_description_clean'] + ' ' +
        train_df['product_bullet_point_clean'] + ' ' +
        train_df['product_brand_clean'] + ' ' +
        train_df['product_color_clean']
    ).str.strip()
    
    test_df['product_text'] = (
        test_df['product_title_clean'] + ' ' +
        test_df['product_description_clean'] + ' ' +
        test_df['product_bullet_point_clean'] + ' ' +
        test_df['product_brand_clean'] + ' ' +
        test_df['product_color_clean']
    ).str.strip()
    
    # добавим фичи
    #пересечения по коэффициенту Жаккара
    def jaccard_similarity(text1, text2):
        set1, set2 = set(text1.split()), set(text2.split())
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union)
    #доля слов из запроса встречается в товаре
    def word_overlap_ratio(query, product):
        query_words = set(w for w in query.split() if len(w) > 2)
        product_words = set(product.split())
        if not query_words:
            return 0.0
        matches = len(query_words & product_words)
        return matches / len(query_words)
    #длины
    def text_length_features(query, product):
        q_len = len(query.split())
        p_len = len(product.split())
        return {
            'query_len': q_len,
            'product_len': p_len,
            'len_ratio': q_len / (p_len + 1)
        }
    
    train_df['title_len'] = train_df['product_title_clean'].apply(lambda x: len(x.split()))
    train_df['bullet_len'] = train_df['product_bullet_point_clean'].apply(lambda x: len(x.split()))
    test_df['title_len'] = test_df['product_title_clean'].apply(lambda x: len(x.split()))
    test_df['bullet_len'] = test_df['product_bullet_point_clean'].apply(lambda x: len(x.split()))
    
    # сопоставление первых токенов
    def first_token_match(query, title):
        query_tokens = query.split()
        title_tokens = title.split()
    
        if not query_tokens or not title_tokens:
            return 0.0
    
        return 1.0 if query_tokens[0] == title_tokens[0] else 0.0
    
    train_df['first_token_match'] = [first_token_match(q, t) for q, t in
                                     zip(train_df['query_clean'], train_df['product_title_clean'])]
    test_df['first_token_match'] = [first_token_match(q, t) for q, t in
                                    zip(test_df['query_clean'], test_df['product_title_clean'])]
    
    train_df['jaccard'] = [jaccard_similarity(q, p) for q, p in
                           zip(train_df['query_clean'], train_df['product_text'])]
    test_df['jaccard'] = [jaccard_similarity(q, p) for q, p in
                         zip(test_df['query_clean'], test_df['product_text'])]
    
    train_df['word_overlap'] = [word_overlap_ratio(q, p) for q, p in
                               zip(train_df['query_clean'], train_df['product_text'])]
    test_df['word_overlap'] = [word_overlap_ratio(q, p) for q, p in
                              zip(test_df['query_clean'], test_df['product_text'])]
    
    
    train_len_features = pd.DataFrame([
        text_length_features(q, p) for q, p in
        zip(train_df['query_clean'], train_df['product_text'])
    ])
    test_len_features = pd.DataFrame([
        text_length_features(q, p) for q, p in
        zip(test_df['query_clean'], test_df['product_text'])
    ])
    
    for col in train_len_features.columns:
        train_df[col] = train_len_features[col]
        test_df[col] = test_len_features[col]
    
    #совпадение бренда или цвета продукта
    def exact_match(query, field):
        if not field or field == 'none':
            return 0.0
        field_words = [w for w in field.split() if len(w) > 2]
        query_words = set(query.split())
        if not field_words:
            return 0.0
        matches = sum(1 for w in field_words if w in query_words)
        return matches / len(field_words)
    
    
    train_df['brand_match'] = [exact_match(q, b) for q, b in
                               zip(train_df['query_clean'], train_df['product_brand_clean'])]
    test_df['brand_match'] = [exact_match(q, b) for q, b in
                             zip(test_df['query_clean'], test_df['product_brand_clean'])]
    
    train_df['color_match'] = [exact_match(q, c) for q, c in
                              zip(train_df['query_clean'], train_df['product_color_clean'])]
    test_df['color_match'] = [exact_match(q, c) for q, c in
                             zip(test_df['query_clean'], test_df['product_color_clean'])]
    
    # фичи на основе упрощенного BM25
    class BM25:
        def __init__(self, documents, k1=1.5, b=0.75):
            self.documents = [doc.split() for doc in documents]
            self.n_docs = len(documents)
            self.avg_doc_len = np.mean([len(doc) for doc in self.documents])
    
            self.k1 = k1
            self.b = b
    
            df = Counter()
            for doc in self.documents:
                df.update(set(doc))
    
            self.idf = {
                word: math.log((self.n_docs - freq + 0.5) / (freq + 0.5) + 1)
                for word, freq in df.items()
            }
    
        def score(self, query, doc_idx):
            query_words = query.split()
            query_tf = Counter(query_words)
    
            doc = self.documents[doc_idx]
            doc_len = len(doc)
            doc_tf = Counter(doc)
    
            score = 0.0
            norm = self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
    
            for word, qf in query_tf.items():
                if word not in doc_tf:
                    continue
    
                tf = doc_tf[word]
                idf = self.idf.get(word, 0.0)
    
                score += idf * (tf * (self.k1 + 1)) / (tf + norm)
            return score
    
    
    
    # BM25
    all_products = pd.concat([train_df['product_text'], test_df['product_text']]).values
    bm25 = BM25(all_products)
    train_df['bm25'] = [bm25.score(q, i) for i, q in enumerate(train_df['query_clean'])]
    test_df['bm25'] = [bm25.score(q, i + len(train_df)) for i, q in enumerate(test_df['query_clean'])]
    
    # BM25 только для заголовка
    all_titles = pd.concat([train_df['product_title_clean'], test_df['product_title_clean']]).values
    bm25_title = BM25(all_titles)
    train_df['bm25_title'] = [bm25_title.score(q, i) for i, q in enumerate(train_df['query_clean'])]
    test_df['bm25_title'] = [bm25_title.score(q, i + len(train_df)) for i, q in enumerate(test_df['query_clean'])]
    
    # BM25 только для описания
    all_descriptions = pd.concat([train_df['product_description_clean'], test_df['product_description_clean']]).values
    bm25_description = BM25(all_descriptions)
    train_df['bm25_description'] = [bm25_description.score(q, i) for i, q in enumerate(train_df['query_clean'])]
    test_df['bm25_description'] = [bm25_description.score(q, i + len(train_df)) for i, q in enumerate(test_df['query_clean'])]
    
    
    feature_cols = ['jaccard', 'word_overlap', 'first_token_match', 'query_len', 'product_len', 'len_ratio', 'title_len',
    'bullet_len', 'brand_match', 'color_match', 'bm25', 'bm25_title', 'bm25_description']
    
    print("Количество фичей в итоге:", len(feature_cols))
    
    
    # делим данные и проверяем модели
    X_train = train_df[feature_cols].copy()
    y_train = train_df['relevance'].values
    group_id_train = train_df['query_id'].values
    
    X_test = test_df[feature_cols].copy()
    
    # задаем NDCG@10
    def calculate_ndcg_at_10(relevance, predictions, query_ids, k=10):
        df = pd.DataFrame({
            'relevance': relevance,
            'prediction': predictions,
            'query_id': query_ids
        })
    
        ndcg_scores = []
        for query_id, group in df.groupby('query_id'):
            rel = group['relevance'].values
            pred = group['prediction'].values
    
            sorted_idx = np.argsort(-pred)
            sorted_rel = rel[sorted_idx][:k]
    
            dcg = np.sum((2 ** sorted_rel - 1) / np.log2(np.arange(2, len(sorted_rel) + 2)))
    
            ideal_rel = np.sort(rel)[::-1][:k]
            idcg = np.sum((2 ** ideal_rel - 1) / np.log2(np.arange(2, len(ideal_rel) + 2)))
    
            if idcg > 0:
                ndcg_scores.append(dcg / idcg)
            else:
                ndcg_scores.append(0.0)
    
        return np.mean(ndcg_scores)
    
    
    n_splits = 10
    gkf = GroupKFold(n_splits=n_splits)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=group_id_train), 1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        group_tr = group_id_train[train_idx]
        group_val = group_id_train[val_idx]
    
        train_pool = Pool(
            data=X_tr,
            label=y_tr,
            group_id=group_tr
        )
    
        val_pool = Pool(
            data=X_val,
            label=y_val,
            group_id=group_val
        )
    
    
        model = CatBoostRanker(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            loss_function='YetiRank',
            eval_metric='NDCG:top=10',
            random_seed=RANDOM_SEED,
            verbose=100,
            early_stopping_rounds=50,
            use_best_model=True
        )
    
        model.fit(train_pool, eval_set=val_pool, verbose=False)
    
        val_pred = model.predict(X_val)
    
        ndcg_score = calculate_ndcg_at_10(y_val, val_pred, group_val)
        cv_scores.append(ndcg_score)
        print(f"Fold {fold}: NDCG@10 = {ndcg_score:.4f}")
    
    print(f"\nMean CV NDCG@10: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    print()
    
    # финальная модель
    train_pool_full = Pool(
        data=X_train,
        label=y_train,
        group_id=group_id_train
    )
    
    final_model = CatBoostRanker(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function='YetiRank',
        eval_metric='NDCG:top=10',
        random_seed=RANDOM_SEED,
        verbose=200
    )
    
    final_model.fit(train_pool_full, verbose=False)
    test_predictions = final_model.predict(X_test)
    
    
    submission = pd.DataFrame({
        'id': test_df['id'],
        'prediction': test_predictions
    })

    create_submission(submission)
    
    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)


if __name__ == "__main__":
    main()
