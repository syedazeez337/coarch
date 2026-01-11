document.addEventListener('DOMContentLoaded', () => {
    const queryInput = document.getElementById('query');
    const languageSelect = document.getElementById('language');
    const searchBtn = document.getElementById('searchBtn');
    const resultsDiv = document.getElementById('results');
    const statusDiv = document.getElementById('status');

    let currentResults = [];

    searchBtn?.addEventListener('click', () => {
        const query = queryInput?.value?.trim();
        if (query) {
            performSearch(query);
        }
    });

    queryInput?.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const query = queryInput.value.trim();
            if (query) {
                performSearch(query);
            }
        }
    });

    function performSearch(query) {
        const language = languageSelect?.value || '';
        const limit = 20;

        showStatus('Searching...');
        resultsDiv.innerHTML = '';

        vscode.postMessage({
            type: 'search',
            query,
            language,
            limit,
        });
    }

    function showStatus(message) {
        statusDiv.textContent = message;
        statusDiv.className = 'status';
    }

    function showError(message) {
        statusDiv.textContent = message;
        statusDiv.className = 'status error';
    }

    window.addEventListener('message', (event) => {
        const message = event.data;

        switch (message.type) {
            case 'results':
                currentResults = message.results;
                displayResults(message.results);
                showStatus(`Found ${message.results.length} results`);
                break;
            case 'error':
                showError(message.message);
                break;
            case 'setQuery':
                if (queryInput) {
                    queryInput.value = message.query;
                }
                break;
        }
    });

    function displayResults(results) {
        if (!resultsDiv) return;

        if (results.length === 0) {
            resultsDiv.innerHTML = '<div class="no-results">No results found</div>';
            return;
        }

        const html = results.map((result, index) => `
            <div class="result" data-index="${index}">
                <div class="result-header">
                    <span class="file-path">${escapeHtml(result.file_path)}</span>
                    <span class="lines">${escapeHtml(result.lines)}</span>
                    <span class="score">${(result.score * 100).toFixed(1)}%</span>
                </div>
                <pre class="code-preview"><code>${escapeHtml(result.code)}</code></pre>
                <div class="result-actions">
                    <button class="action-btn open-btn">Open</button>
                    <button class="action-btn copy-btn">Copy</button>
                </div>
            </div>
        `).join('');

        resultsDiv.innerHTML = html;

        // Add event listeners
        resultsDiv.querySelectorAll('.open-btn').forEach((btn, index) => {
            btn.addEventListener('click', () => {
                const result = currentResults[index];
                vscode.postMessage({
                    type: 'open',
                    filePath: result.file_path,
                    startLine: parseInt(result.lines.split('-')[0]),
                });
            });
        });

        resultsDiv.querySelectorAll('.copy-btn').forEach((btn, index) => {
            btn.addEventListener('click', () => {
                const result = currentResults[index];
                vscode.postMessage({
                    type: 'copy',
                    code: result.code,
                });
            });
        });
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
});
