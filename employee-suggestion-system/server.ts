import http from 'http';
import { loadSuggestionData, getOutputFields } from './data-loader';
import { SuggestionIndex } from './embeddings';
import { config } from './config';
import { predict } from './index';
import { appendData, getDataSummary } from './data-import';
import { OutputFields, Suggestion } from './types';

let index: SuggestionIndex | null = null;
let cachedData: Suggestion[] = [];

async function ensureIndex() {
  if (!index) {
    cachedData = await loadSuggestionData();
    index = new SuggestionIndex();
    await index.buildIndex(cachedData);
    console.log('Index ready!');
  }
}

const htmlPage = `
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>员工合理化建议系统</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }
    .container { max-width: 1200px; margin: 0 auto; }
    .header { background: white; padding: 20px 30px; border-radius: 10px 10px 0 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
    .header h1 { color: #333; font-size: 24px; }
    .header p { color: #666; margin-top: 5px; }
    .main { background: white; padding: 30px; border-radius: 0 0 10px 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
    .tabs { display: flex; gap: 10px; margin-bottom: 20px; border-bottom: 2px solid #eee; padding-bottom: 10px; }
    .tab { padding: 10px 20px; cursor: pointer; border: none; background: none; font-size: 16px; color: #666; border-radius: 5px; transition: all 0.3s; }
    .tab:hover { background: #f5f5f5; }
    .tab.active { background: #667eea; color: white; }
    .tab-content { display: none; }
    .tab-content.active { display: block; }
    .form-group { margin-bottom: 20px; }
    .form-group label { display: block; margin-bottom: 8px; font-weight: 600; color: #333; }
    .form-group input, .form-group textarea, .form-group select { width: 100%; padding: 12px; border: 1px solid #ddd; border-radius: 5px; font-size: 14px; transition: border-color 0.3s; }
    .form-group input:focus, .form-group textarea:focus, .form-group select:focus { outline: none; border-color: #667eea; }
    .form-group textarea { min-height: 120px; resize: vertical; }
    .row { display: flex; gap: 20px; flex-wrap: wrap; }
    .col { flex: 1; min-width: 250px; }
    .btn { padding: 12px 30px; border: none; border-radius: 5px; font-size: 16px; cursor: pointer; transition: all 0.3s; }
    .btn-primary { background: #667eea; color: white; }
    .btn-primary:hover { background: #5568d3; transform: translateY(-2px); box-shadow: 0 4px 10px rgba(102, 126, 234, 0.4); }
    .btn-secondary { background: #6c757d; color: white; }
    .btn-secondary:hover { background: #5a6268; }
    .result { background: #f8f9fa; padding: 20px; border-radius: 5px; margin-top: 20px; border-left: 4px solid #667eea; }
    .result h3 { color: #333; margin-bottom: 15px; }
    .result-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; }
    .result-item { background: white; padding: 15px; border-radius: 5px; border: 1px solid #eee; }
    .result-item label { font-size: 12px; color: #666; display: block; margin-bottom: 5px; }
    .result-item span { font-size: 14px; color: #333; font-weight: 500; }
    .similarity { background: #e8f5e9; padding: 10px; border-radius: 5px; margin-bottom: 15px; color: #2e7d32; }
    .loading { text-align: center; padding: 40px; color: #666; }
    .spinner { border: 3px solid #f3f3f3; border-top: 3px solid #667eea; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto 15px; }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    .error { background: #ffebee; color: #c62828; padding: 15px; border-radius: 5px; margin-top: 15px; }
    .stats { background: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
    .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; }
    .stat-item { text-align: center; padding: 15px; background: white; border-radius: 5px; }
    .stat-value { font-size: 28px; font-weight: bold; color: #667eea; }
    .stat-label { font-size: 12px; color: #666; margin-top: 5px; }
    .file-upload { border: 2px dashed #ddd; padding: 40px; text-align: center; border-radius: 5px; cursor: pointer; transition: all 0.3s; }
    .file-upload:hover { border-color: #667eea; background: #f8f9ff; }
    .file-upload input { display: none; }
    .file-name { margin-top: 10px; color: #666; }
    .history-table { width: 100%; border-collapse: collapse; margin-top: 15px; }
    .history-table th, .history-table td { padding: 12px; text-align: left; border-bottom: 1px solid #eee; }
    .history-table th { background: #f8f9fa; font-weight: 600; color: #333; }
    .history-table tr:hover { background: #f8f9ff; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>员工合理化建议智能预测系统</h1>
      <p>基于历史数据学习的智能推荐系统</p>
    </div>
    <div class="main">
      <div class="tabs">
        <button class="tab active" onclick="switchTab('predict')">预测</button>
        <button class="tab" onclick="switchTab('import')">数据导入</button>
        <button class="tab" onclick="switchTab('stats')">数据统计</button>
      </div>

      <div id="predict" class="tab-content active">
        <form id="predictForm">
          <div class="row">
            <div class="col">
              <div class="form-group">
                <label>员工GID *</label>
                <input type="text" id="gid" placeholder="请输入员工GID" required>
              </div>
            </div>
          </div>
          <div class="form-group">
            <label>问题描述 *</label>
            <textarea id="description" placeholder="请详细描述发现的问题或改进点..." required></textarea>
          </div>
          <div class="form-group">
            <label>建议方案 *</label>
            <textarea id="suggestion" placeholder="请详细描述您的改进建议..." required></textarea>
          </div>
          <button type="submit" class="btn btn-primary">开始预测</button>
        </form>
        <div id="predictResult"></div>
      </div>

      <div id="import" class="tab-content">
        <div class="file-upload" onclick="document.getElementById('fileInput').click()">
          <input type="file" id="fileInput" accept=".xlsx,.xls">
          <div style="font-size: 40px; color: #667eea;">+</div>
          <p>点击选择Excel文件或拖拽到此处</p>
          <p class="file-name" id="fileName"></p>
        </div>
        <div style="margin-top: 20px;">
          <button class="btn btn-primary" onclick="importData()">导入数据</button>
          <button class="btn btn-secondary" onclick="rebuildIndex()">重建索引</button>
        </div>
        <div id="importResult"></div>
      </div>

      <div id="stats" class="tab-content">
        <div id="statsContent"></div>
      </div>
    </div>
  </div>

  <script>
    let selectedFile = null;

    document.getElementById('fileInput').addEventListener('change', function(e) {
      selectedFile = e.target.files[0];
      document.getElementById('fileName').textContent = selectedFile ? selectedFile.name : '';
    });

    function switchTab(tabId) {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      document.querySelector('[onclick="switchTab(\\'' + tabId + '\\')"]').classList.add('active');
      document.getElementById(tabId).classList.add('active');
      if (tabId === 'stats') loadStats();
    }

    document.getElementById('predictForm').addEventListener('submit', async function(e) {
      e.preventDefault();
      const resultDiv = document.getElementById('predictResult');
      resultDiv.innerHTML = '<div class="loading"><div class="spinner"></div><p>正在分析并预测...</p></div>';

      try {
        const response = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            description: document.getElementById('description').value,
            suggestion: document.getElementById('suggestion').value,
            gid: document.getElementById('gid').value
          })
        });

        const result = await response.json();
        
        if (!response.ok) {
          throw new Error(result.error || '预测失败');
        }

        let html = '<div class="similarity">匹配度: ' + (result.similarity * 100).toFixed(1) + '%</div>';
        html += '<div class="result"><h3>预测结果</h3><div class="result-grid">';
        
        const fields = ${JSON.stringify(OutputFields)};
        fields.forEach(field => {
          html += '<div class="result-item"><label>' + field + '</label><span>' + (result[field] || '-') + '</span></div>';
        });
        
        html += '</div></div>';
        resultDiv.innerHTML = html;
      } catch (err) {
        resultDiv.innerHTML = '<div class="error">' + err.message + '</div>';
      }
    });

    async function importData() {
      const resultDiv = document.getElementById('importResult');
      resultDiv.innerHTML = '<div class="loading"><div class="spinner"></div><p>正在读取桌面文件并导入数据...</p></div>';

      try {
        const response = await fetch('/api/import', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'import' })
        });

        const result = await response.json();
        
        if (result.success) {
          resultDiv.innerHTML = '<div class="similarity">成功导入 ' + result.imported + ' 条数据</div>';
        } else {
          resultDiv.innerHTML = '<div class="error">导入失败: ' + (result.error || result.errors.join(', ')) + '</div>';
        }
      } catch (err) {
        resultDiv.innerHTML = '<div class="error">' + err.message + '</div>';
      }
    }

    async function rebuildIndex() {
      const resultDiv = document.getElementById('importResult');
      resultDiv.innerHTML = '<div class="loading"><div class="spinner"></div><p>正在重建索引...</p></div>';

      try {
        const response = await fetch('/api/rebuild-index', { method: 'POST' });
        const result = await response.json();
        
        resultDiv.innerHTML = '<div class="similarity">索引重建成功! 共 ' + result.total + ' 条记录</div>';
      } catch (err) {
        resultDiv.innerHTML = '<div class="error">' + err.message + '</div>';
      }
    }

    async function loadStats() {
      try {
        const response = await fetch('/api/stats');
        const stats = await response.json();
        
        let html = '<div class="stats"><div class="stats-grid">';
        html += '<div class="stat-item"><div class="stat-value">' + stats.total + '</div><div class="stat-label">总记录数</div></div>';
        html += '<div class="stat-item"><div class="stat-value">' + stats.departments.length + '</div><div class="stat-label">部门数</div></div>';
        html += '<div class="stat-item"><div class="stat-value">' + stats.laborTypes.length + '</div><div class="stat-label">用工类型</div></div>';
        html += '<div class="stat-item"><div class="stat-value">' + stats.itemTypes.length + '</div><div class="stat-label">建议类型</div></div>';
        html += '</div></div>';
        
        html += '<h4>部门分布</h4><table class="history-table"><tr><th>部门</th></tr>';
        stats.departments.forEach(d => {
          html += '<tr><td>' + d + '</td></tr>';
        });
        html += '</table>';
        
        document.getElementById('statsContent').innerHTML = html;
      } catch (err) {
        document.getElementById('statsContent').innerHTML = '<div class="error">' + err.message + '</div>';
      }
    }

    loadStats();
  </script>
</body>
</html>
`;

const server = http.createServer(async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.writeHead(200);
    res.end();
    return;
  }

  if (req.url === '/' || req.url === '/index.html') {
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    res.end(htmlPage);
    return;
  }

  if (req.url === '/api/stats' && req.method === 'GET') {
    try {
      const data = await loadSuggestionData();
      const summary = getDataSummary(data);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(summary));
    } catch (e) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: String(e) }));
    }
    return;
  }

  if (req.url === '/api/rebuild-index' && req.method === 'POST') {
    try {
      index = null;
      cachedData = await loadSuggestionData();
      index = new SuggestionIndex();
      await index.buildIndex(cachedData);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ success: true, total: cachedData.length }));
    } catch (e) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: String(e) }));
    }
    return;
  }

  if (req.url === '/api/predict' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', async () => {
      try {
        const { description, suggestion, gid } = JSON.parse(body);
        
        await ensureIndex();
        
        const query = `${description} ${suggestion} ${gid}`;
        const results = await index!.searchAsync(query, 1);
        
        if (results.length === 0) {
          throw new Error('No similar records found');
        }

        const output = getOutputFields(results[0].suggestion);
        
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          ...output,
          similarity: results[0].similarity
        }));
      } catch (e) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: String(e) }));
      }
    });
    return;
  }

  if (req.url === '/api/import' && req.method === 'POST') {
    const fs = await import('fs');

    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', async () => {
      try {
        const desktopPath = 'C:\\Users\\364328\\Desktop\\2025 ESS simple.xlsx';
        
        if (!fs.existsSync(desktopPath)) {
          throw new Error('Desktop file not found: ' + desktopPath);
        }

        const result = await appendData(desktopPath, cachedData);

        index = null;
        await ensureIndex();

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(result));
      } catch (e) {
        console.error('Import error:', e);
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: String(e), success: false, imported: 0, errors: [String(e)] }));
      }
    });
    return;
  }

  res.writeHead(404);
  res.end('Not Found');
});

const PORT = 3000;
server.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}/`);
});
