from flask import Flask, render_template, request, jsonify
import csv
import os
from datetime import datetime
from filelock import FileLock

app = Flask(__name__)
CSV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'surveys.csv')
LOCK_FILE = CSV_FILE + '.lock'

def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['employee_id', 'score', 'submitted_at'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/check', methods=['POST'])
def check():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({'error': 'Invalid JSON'}), 400
    
    employee_id = data.get('employee_id', '').strip()
    
    if not employee_id:
        return jsonify({'error': '工号不能为空'}), 400
    
    lock = FileLock(LOCK_FILE, timeout=10)
    with lock:
        if os.path.exists(CSV_FILE):
            with open(CSV_FILE, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['employee_id'] == employee_id:
                        return jsonify({'exists': True, 'score': row['score']})
    
    return jsonify({'exists': False})

@app.route('/api/submit', methods=['POST'])
def submit():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({'error': 'Invalid JSON'}), 400
    
    employee_id = data.get('employee_id', '').strip()
    score = data.get('score')
    
    if not employee_id:
        return jsonify({'error': '工号不能为空'}), 400
    
    try:
        score_int = int(score)
        if not (1 <= score_int <= 10):
            return jsonify({'error': '评分必须在1-10之间'}), 400
    except (TypeError, ValueError):
        return jsonify({'error': '评分必须在1-10之间'}), 400
    
    lock = FileLock(LOCK_FILE, timeout=10)
    with lock:
        if os.path.exists(CSV_FILE):
            with open(CSV_FILE, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['employee_id'] == employee_id:
                        return jsonify({'error': '您已提交过'}), 400
        
        with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([employee_id, score_int, datetime.now().isoformat()])
    
    return jsonify({'success': True, 'score': score_int})

if __name__ == '__main__':
    init_csv()
    app.run(debug=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true', port=5000)
