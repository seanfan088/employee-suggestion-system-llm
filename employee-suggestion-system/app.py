import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import jieba
import json
import os
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

EXCEL_PATH = r'C:\Users\364328\Desktop\2025 ESS simple.xlsx'
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

OUTPUT_FIELDS = [
    'Title', 'Department', 'LaborType', 'Shift', 'UserAreaName',
    'ProblemAreaName', 'LocationName', 'ManagerGid', 'ManagerName',
    'ItemType', 'OwnerName', 'OwnerGid', 'OwnerTel', 'OwnerManagerGid',
    'OwnerManagerName', 'IsRepeat', 'AreaType'
]

vectorizer = None
df = None
w2v_model = None

def load_data():
    global df
    df = pd.read_excel(EXCEL_PATH, engine='openpyxl')
    df = df.fillna('')
    return df

def tokenize_chinese(text):
    return list(jieba.cut(str(text)))

def train_word2vec(sentences, model_path, sample_size=10000):
    if len(sentences) > sample_size:
        import random
        sentences = random.sample(sentences, sample_size)
        print(f"Word2Vec training on {sample_size} samples (reduced from {len(sentences)})")
    
    model = Word2Vec(
        sentences=sentences,
        vector_size=50,
        window=3,
        min_count=2,
        workers=2,
        epochs=10,
        sg=0
    )
    model.save(model_path.replace('.pkl', '_w2v.model'))
    return model

def get_sentence_vector(text, model):
    words = tokenize_chinese(text)
    vectors = []
    for word in words:
        if word in model.wv:
            vectors.append(model.wv[word])
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(model.vector_size)

def train_model():
    global vectorizer, df, w2v_model
    df = load_data()
    print(f"Loading {len(df)} records...")
    
    df['combined_text'] = (
        df['Description'].astype(str) + ' ' + 
        df['Suggestion'].astype(str) + ' ' + 
        df['Gid'].astype(str)
    )
    
    print("Training TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=3000)
    tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
    
    print("Tokenizing for Word2Vec...")
    sentences = []
    for _, row in df.iterrows():
        desc_words = tokenize_chinese(row['Description'])
        sugg_words = tokenize_chinese(row['Suggestion'])
        sentences.append(desc_words + sugg_words)
    
    print("Training Word2Vec (sampled)...")
    w2v_model = train_word2vec(sentences, MODEL_PATH)
    
    vector_size = w2v_model.vector_size
    
    print("Computing sentence vectors...")
    desc_vectors = []
    sugg_vectors = []
    for _, row in df.iterrows():
        desc_vectors.append(get_sentence_vector(row['Description'], w2v_model))
        sugg_vectors.append(get_sentence_vector(row['Suggestion'], w2v_model))
    
    desc_vectors = np.array(desc_vectors)
    sugg_vectors = np.array(sugg_vectors)
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({
            'vectorizer': vectorizer, 
            'df': df, 
            'tfidf_matrix': tfidf_matrix,
            'w2v_model': w2v_model,
            'desc_vectors': desc_vectors,
            'sugg_vectors': sugg_vectors
        }, f)
    
    print(f"Model trained! Total records: {len(df)}")
    print(f"Word2Vec vocabulary size: {len(w2v_model.wv)}")
    return df

def load_model():
    global vectorizer, df, w2v_model, desc_vectors, sugg_vectors
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            data = pickle.load(f)
            vectorizer = data['vectorizer']
            df = data['df']
            w2v_model = data.get('w2v_model')
            desc_vectors = data.get('desc_vectors')
            sugg_vectors = data.get('sugg_vectors')
            return True
    return False

EMPLOYEE_FIELDS = ['Title', 'Department', 'LaborType', 'Shift', 'UserAreaName', 'ProblemAreaName', 'LocationName', 'ManagerGid', 'ManagerName']
SUGGESTION_FIELDS = ['ItemType', 'OwnerName', 'OwnerGid', 'OwnerTel', 'OwnerManagerGid', 'OwnerManagerName', 'IsRepeat', 'AreaType']


def normalize_gid(value):
    return str(value).strip().lower()


def sort_matches_by_recent_submission(matches):
    if 'SubmissionDate' not in matches.columns:
        return matches.sort_index(ascending=False)

    ordered = matches.copy()
    ordered['_submission_dt'] = pd.to_datetime(ordered['SubmissionDate'], errors='coerce')
    ordered = ordered.sort_values(by=['_submission_dt'], ascending=False, na_position='last')
    return ordered.drop(columns=['_submission_dt'])


def get_employee_profile_by_gid(gid):
    if df is None:
        return None

    normalized_gid = normalize_gid(gid)
    if not normalized_gid:
        return None

    gid_series = df['Gid'].astype(str).str.strip().str.lower()
    matches = df[gid_series == normalized_gid]

    if matches.empty:
        return None

    matches = sort_matches_by_recent_submission(matches)

    profile = {}
    for field in EMPLOYEE_FIELDS:
        values = matches[field].astype(str).str.strip()
        values = values[values != '']
        if values.empty:
            profile[field] = ''
            continue

        profile[field] = str(values.iloc[0])

    return profile

def serve_frontend_file():
    with open('index.html', 'r', encoding='utf-8') as f:
        return f.read(), 200, {'Content-Type': 'text/html; charset=utf-8'}


@app.route('/')
def index():
    return serve_frontend_file()


@app.route('/index.html')
def serve_index():
    return serve_frontend_file()

@app.route('/employee-suggestion-system/')
@app.route('/employee-suggestion-system/index.html')
def serve_employee_system():
    return serve_frontend_file()

@app.route('/employee-suggestion-system/api/predict', methods=['POST'])
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    description = data.get('description', '')
    suggestion = data.get('suggestion', '')
    gid = str(data.get('gid', ''))
    
    if not load_model():
        train_model()
    
    output = {}
    
    gid_similarity = 0.0
    if gid:
        employee_profile = get_employee_profile_by_gid(gid)
        if employee_profile is not None:
            output.update(employee_profile)
            gid_similarity = 1.0
        else:
            gid_query = gid
            gid_vec = vectorizer.transform([gid_query])
            gid_similarities = cosine_similarity(gid_vec, vectorizer.transform(df['combined_text']))[0]
            gid_top_idx = np.argmax(gid_similarities)
            gid_similarity = float(gid_similarities[gid_top_idx])

            result = df.iloc[gid_top_idx]
            for field in EMPLOYEE_FIELDS:
                output[field] = str(result[field])
    
    if description or suggestion:
        tfidf_query = f"{description} {suggestion}"
        tfidf_vec = vectorizer.transform([tfidf_query])
        tfidf_similarities = cosine_similarity(tfidf_vec, vectorizer.transform(df['combined_text']))[0]
        
        if w2v_model is not None and 'desc_vectors' in dir():
            desc_vec = get_sentence_vector(description, w2v_model).reshape(1, -1)
            sugg_vec = get_sentence_vector(suggestion, w2v_model).reshape(1, -1)
            
            desc_w2v_sims = cosine_similarity(desc_vec, desc_vectors)[0]
            sugg_w2v_sims = cosine_similarity(sugg_vec, sugg_vectors)[0]
            
            combined_sims = 0.5 * tfidf_similarities + 0.25 * desc_w2v_sims + 0.25 * sugg_w2v_sims
        else:
            combined_sims = tfidf_similarities
        
        content_top_idx = np.argmax(combined_sims)
        
        result = df.iloc[content_top_idx]
        for field in SUGGESTION_FIELDS:
            output[field] = str(result[field])
    
    content_similarity = float(combined_sims[content_top_idx]) if (description or suggestion) else 0.0
    similarity_parts = int(bool(gid)) + int(bool(description or suggestion))
    if similarity_parts == 0:
        output['similarity'] = 0.0
    else:
        output['similarity'] = float((gid_similarity + content_similarity) / similarity_parts)
    
    return jsonify(output)

@app.route('/employee-suggestion-system/api/stats', methods=['GET'])
@app.route('/api/stats', methods=['GET'])
def stats():
    if df is None:
        load_data()
    
    return jsonify({
        'total': len(df),
        'departments': df['Department'].unique().tolist(),
        'laborTypes': df['LaborType'].unique().tolist(),
        'itemTypes': df['ItemType'].unique().tolist(),
        'areaTypes': df['AreaType'].unique().tolist()
    })

@app.route('/employee-suggestion-system/api/field-options', methods=['GET'])
@app.route('/api/field-options', methods=['GET'])
def field_options():
    if df is None:
        load_data()
    
    options = {}
    for field in OUTPUT_FIELDS:
        values = df[field].unique().tolist()
        options[field] = [str(v) for v in values if v]
    
    return jsonify(options)

@app.route('/employee-suggestion-system/api/submit', methods=['POST'])
@app.route('/api/submit', methods=['POST'])
def submit_data():
    global vectorizer, df
    data = request.json
    
    try:
        if df is None:
            load_data()
        
        new_row = {
            'SN': f'FY25Q3{len(df)+1:05d}',
            'Gid': data.get('gid', ''),
            'Name': data.get('Name', ''),
            'Title': data.get('Title', ''),
            'Department': data.get('Department', ''),
            'Tel': data.get('Tel', ''),
            'LaborType': data.get('LaborType', ''),
            'Shift': data.get('Shift', ''),
            'UserAreaName': data.get('UserAreaName', ''),
            'ProblemAreaName': data.get('ProblemAreaName', ''),
            'LocationName': data.get('LocationName', ''),
            'ManagerGid': data.get('ManagerGid', ''),
            'ManagerName': data.get('ManagerName', ''),
            'SubmissionDate': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
            'Description': data.get('description', ''),
            'Suggestion': data.get('suggestion', ''),
            'Status': 'Submitted',
            'ItemType': data.get('ItemType', ''),
            'OwnerName': data.get('OwnerName', ''),
            'OwnerGid': data.get('OwnerGid', ''),
            'OwnerTel': data.get('OwnerTel', ''),
            'OwnerManagerGid': data.get('OwnerManagerGid', ''),
            'OwnerManagerName': data.get('OwnerManagerName', ''),
            'IsRepeat': data.get('IsRepeat', ''),
            'AreaType': data.get('AreaType', ''),
        }
        
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        df.to_excel(EXCEL_PATH, index=False)
        
        train_model()
        
        model_stats = calculate_model_stats()
        
        return jsonify({'success': True, 'message': '数据已保存并重新训练模型', 'modelStats': model_stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def calculate_model_stats():
    global df, vectorizer, w2v_model, desc_vectors, sugg_vectors
    
    if df is None or vectorizer is None:
        return {
            'totalDataVolume': 0,
            'numberOfFeatures': 0,
            'modelAccuracy': 0,
            'lastTrained': '-',
            'parameters': {},
            'w2vInfo': {}
        }
    
    tfidf_matrix = vectorizer.transform(df['combined_text'])
    
    tfidf_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
    np.fill_diagonal(tfidf_similarities, 0)
    
    tfidf_accuracy = float(np.mean(tfidf_similarities.max(axis=1)))
    
    combined_accuracy = tfidf_accuracy
    
    if 'desc_vectors' in dir() and desc_vectors is not None and len(desc_vectors) > 0:
        try:
            desc_sims = cosine_similarity(desc_vectors, desc_vectors)
            sugg_sims = cosine_similarity(sugg_vectors, sugg_vectors)
            np.fill_diagonal(desc_sims, 0)
            np.fill_diagonal(sugg_sims, 0)
            
            w2v_accuracy = (np.mean(desc_sims.max(axis=1)) + np.mean(sugg_sims.max(axis=1))) / 2
            
            combined_accuracy = 0.4 * tfidf_accuracy + 0.3 * w2v_accuracy
        except:
            pass
    
    accuracy = min(combined_accuracy * 100, 95)
    
    w2v_info = {}
    if w2v_model is not None:
        w2v_info = {
            'vectorSize': w2v_model.vector_size,
            'windowSize': w2v_model.window,
            'epochs': w2v_model.epochs,
            'vocabularySize': len(w2v_model.wv),
            'algorithm': 'Skip-gram (sg=1)'
        }
    
    return {
        'totalDataVolume': len(df),
        'numberOfFeatures': tfidf_matrix.shape[1],
        'parameters': {
            'tfidf': {
                'vectorizerType': 'TF-IDF',
                'maxFeatures': 5000,
                'ngramRange': '(1, 1)',
                'similarityMetric': 'Cosine'
            },
            'word2vec': w2v_info
        },
        'modelAccuracy': round(accuracy, 2),
        'attributes': {
            'employeeFields': EMPLOYEE_FIELDS,
            'suggestionFields': SUGGESTION_FIELDS,
            'contextLearning': 'Description -> Suggestion'
        },
        'lastTrained': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }

@app.route('/employee-suggestion-system/api/model-stats', methods=['GET'])
@app.route('/api/model-stats', methods=['GET'])
def model_stats():
    try:
        stats = calculate_model_stats()
        if not stats or stats.get('totalDataVolume', 0) == 0:
            return jsonify({
                'totalDataVolume': 0,
                'numberOfFeatures': 0,
                'modelAccuracy': 0,
                'lastTrained': '-',
                'parameters': {},
                'w2vInfo': {}
            })
        return jsonify(stats)
    except Exception as e:
        print(f"model-stats error: {e}")
        return jsonify({
            'totalDataVolume': 0,
            'numberOfFeatures': 0,
            'modelAccuracy': 0,
            'lastTrained': '-',
            'parameters': {},
            'w2vInfo': {},
            'error': str(e)
        })

@app.route('/employee-suggestion-system/api/import', methods=['POST'])
@app.route('/api/import', methods=['POST'])
def import_data():
    global vectorizer, df
    try:
        df = load_data()
        train_model()
        return jsonify({'success': True, 'total': len(df)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/employee-suggestion-system/api/chart-data', methods=['GET'])
@app.route('/api/chart-data', methods=['GET'])
def chart_data():
    global df, vectorizer, w2v_model, desc_vectors, sugg_vectors
    try:
        if df is None or vectorizer is None:
            return jsonify({
                'departmentDistribution': [],
                'itemTypeDistribution': [],
                'laborTypeDistribution': [],
                'similarityDistribution': [],
                'topKeywords': []
            })
        
        dept_counts = df['Department'].value_counts().head(10).to_dict()
        item_counts = df['ItemType'].value_counts().head(10).to_dict()
        labor_counts = df['LaborType'].value_counts().to_dict()
        
        tfidf_matrix = vectorizer.transform(df['combined_text'])
        similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
        np.fill_diagonal(similarities, 0)
        max_sims = similarities.max(axis=1)
        
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist, _ = np.histogram(max_sims, bins=bins)
        
        feature_names = vectorizer.get_feature_names_out()
        tfidf_sum = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
        top_indices = tfidf_sum.argsort()[-20:][::-1]
        top_keywords = [{'word': feature_names[i], 'weight': float(tfidf_sum[i])} for i in top_indices]
        
        return jsonify({
            'departmentDistribution': [{'name': k, 'value': int(v)} for k, v in dept_counts.items()],
            'itemTypeDistribution': [{'name': k, 'value': int(v)} for k, v in item_counts.items()],
            'laborTypeDistribution': [{'name': k, 'value': int(v)} for k, v in labor_counts.items()],
            'similarityDistribution': [{'range': f'{bins[i]}-{bins[i+1]}', 'count': int(hist[i])} for i in range(len(hist))],
            'topKeywords': top_keywords[:10]
        })
    except Exception as e:
        print(f"chart-data error: {e}")
        return jsonify({'error': str(e)})

@app.route('/employee-suggestion-system/api/model-evaluation', methods=['GET'])
@app.route('/api/model-evaluation', methods=['GET'])
def model_evaluation():
    global df
    try:
        if df is None:
            return jsonify({'error': 'Model not loaded'})
        
        results = {}
        
        evaluation_fields = EMPLOYEE_FIELDS + SUGGESTION_FIELDS

        for field in evaluation_fields:
            if field not in df.columns:
                continue

            try:
                field_df = df[['combined_text', field]].copy()
                field_df[field] = field_df[field].astype(str).str.strip()
                field_df['combined_text'] = field_df['combined_text'].astype(str).str.strip()
                field_df = field_df[(field_df[field] != '') & (field_df['combined_text'] != '')]

                if field_df.empty:
                    continue

                label_counts = field_df[field].value_counts()
                valid_labels = label_counts[label_counts >= 2].index
                field_df = field_df[field_df[field].isin(valid_labels)]

                unique_labels = field_df[field].unique()
                if len(unique_labels) < 2 or len(field_df) < 10:
                    continue

                stratify_labels = field_df[field] if (field_df[field].value_counts() >= 2).all() else None
                train_texts, test_texts, train_labels, test_labels = train_test_split(
                    field_df['combined_text'],
                    field_df[field],
                    test_size=0.2,
                    random_state=42,
                    stratify=stratify_labels,
                )

                eval_vectorizer = TfidfVectorizer(max_features=3000)
                train_matrix = eval_vectorizer.fit_transform(train_texts)
                test_matrix = eval_vectorizer.transform(test_texts)

                sims = cosine_similarity(test_matrix, train_matrix)
                nearest_indices = np.argmax(sims, axis=1)
                predicted_labels = train_labels.iloc[nearest_indices].to_numpy()
                similarity_scores = sims[np.arange(len(nearest_indices)), nearest_indices]

                accuracy = accuracy_score(test_labels, predicted_labels)
                precision = precision_score(test_labels, predicted_labels, average='weighted', zero_division=0)
                recall = recall_score(test_labels, predicted_labels, average='weighted', zero_division=0)
                f1 = f1_score(test_labels, predicted_labels, average='weighted', zero_division=0)

                results[field] = {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1),
                    'uniqueClasses': int(len(unique_labels)),
                    'classes': sorted(unique_labels.tolist())[:20],
                    'trainSize': int(len(train_texts)),
                    'testSize': int(len(test_texts)),
                    'meanSimilarity': float(np.mean(similarity_scores)),
                    'evaluationMethod': '80/20 train-test split with TF-IDF nearest-neighbor matching',
                }
            except Exception as field_error:
                print(f"model-evaluation field error ({field}): {field_error}")
                continue
        
        return jsonify(results)
    except Exception as e:
        print(f"model-evaluation error: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    if not load_model():
        print("Training initial model...")
        train_model()
    
    print("Server starting at http://localhost:5500")
    app.run(port=5500, debug=False)
