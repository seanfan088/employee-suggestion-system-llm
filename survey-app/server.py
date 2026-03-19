from flask import Flask, render_template, request, jsonify
import csv
import os
from datetime import datetime

app = Flask(__name__)
CSV_FILE = 'surveys.csv'

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
    data = request.json
    employee_id = data.get('employee_id', '').strip()
    
    if not employee_id:
        return jsonify({'error': '工号不能为空'}), 400
    
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['employee_id'] == employee_id:
                    return jsonify({'exists': True, 'score': row['score']})
    
    return jsonify({'exists': False})

@app.route('/api/submit', methods=['POST'])
def submit():
    data = request.json
    employee_id = data.get('employee_id', '').strip()
    score = data.get('score')
    
    if not employee_id:
        return jsonify({'error': '工号不能为空'}), 400
    
    if not score or not (1 <= int(score) <= 10):
        return jsonify({'error': '评分必须在1-10之间'}), 400
    
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['employee_id'] == employee_id:
                    return jsonify({'error': '您已提交过'}), 400
    
    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([employee_id, score, datetime.now().isoformat()])
    
    return jsonify({'success': True, 'score': score})

if __name__ == '__main__':
    init_csv()
    app.run(debug=True, port=5000)
