import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';

export class SearchPanel {
    public static currentPanel: vscode.WebviewPanel | undefined;
    private readonly _panel: vscode.WebviewPanel;
    private readonly _extensionUri: vscode.Uri;
    private _disposables: vscode.Disposable[] = [];

    private constructor(panel: vscode.WebviewPanel, extensionUri: vscode.Uri, initialQuery?: string) {
        this._panel = panel;
        this._extensionUri = extensionUri;

        this._panel.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri],
        };

        this._panel.webview.html = this._getHtmlForWebview(initialQuery);

        this._panel.onDidDispose(() => this.dispose(), null, this._disposables);

        this._panel.webview.onDidReceiveMessage(
            async (message) => {
                switch (message.type) {
                    case 'search':
                        await this._performSearch(message.query, message.language, message.limit);
                        break;
                    case 'open':
                        this._openFile(message.filePath, message.startLine);
                        break;
                    case 'copy':
                        await vscode.env.clipboard.writeText(message.code);
                        break;
                }
            },
            null,
            this._disposables
        );
    }

    public static createOrShow(extensionUri: vscode.Uri, initialQuery?: string) {
        const column = vscode.window.activeTextEditor?.viewColumn || vscode.ViewColumn.One;

        if (SearchPanel.currentPanel) {
            SearchPanel.currentPanel.reveal(column);
            if (initialQuery) {
                SearchPanel.currentPanel._panel.webview.postMessage({
                    type: 'setQuery',
                    query: initialQuery,
                });
            }
            return;
        }

        const panel = vscode.window.createWebviewPanel(
            'coarchSearch',
            'Coarch Search',
            column,
            { enableScripts: true }
        );

        SearchPanel.currentPanel = new SearchPanel(panel, extensionUri, initialQuery);
    }

    private async _performSearch(query: string, language?: string, limit: number = 20) {
        try {
            const config = vscode.workspace.getConfiguration('coarch');
            const serverUrl = config.get('serverUrl') || 'http://localhost:8000';

            const response = await fetch(`${serverUrl}/search`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, language, limit }),
            });

            if (response.ok) {
                const results = await response.json();
                this._panel.webview.postMessage({
                    type: 'results',
                    results: results,
                });
            } else {
                this._panel.webview.postMessage({
                    type: 'error',
                    message: 'Search failed',
                });
            }
        } catch (error) {
            this._panel.webview.postMessage({
                type: 'error',
                message: `Error: ${error}`,
            });
        }
    }

    private _openFile(filePath: string, startLine: number) {
        const uri = vscode.Uri.file(filePath);
        vscode.window.showTextDocument(uri, {
            selection: new vscode.Range(startLine - 1, 0, startLine - 1, 0),
            preview: true,
        });
    }

    private _getHtmlForWebview(initialQuery?: string): string {
        const scriptUri = this._panel.webview.asWebviewUri(
            vscode.Uri.joinPath(this._extensionUri, 'media', 'main.js')
        );
        const styleUri = this._panel.webview.asWebviewUri(
            vscode.Uri.joinPath(this._extensionUri, 'media', 'styles.css')
        );

        return `<!DOCTYPE html>
            <html>
            <head>
                <link rel="stylesheet" href="${styleUri}">
            </head>
            <body>
                <div class="container">
                    <div class="search-box">
                        <input type="text" id="query" placeholder="Search code..." value="${initialQuery || ''}">
                        <select id="language">
                            <option value="">All Languages</option>
                            <option value="python">Python</option>
                            <option value="javascript">JavaScript</option>
                            <option value="typescript">TypeScript</option>
                            <option value="java">Java</option>
                            <option value="cpp">C++</option>
                            <option value="go">Go</option>
                            <option value="rust">Rust</option>
                        </select>
                        <button id="searchBtn">Search</button>
                    </div>
                    <div id="status"></div>
                    <div id="results"></div>
                </div>
                <script src="${scriptUri}"></script>
            </body>
            </html>`;
    }

    public dispose() {
        SearchPanel.currentPanel = undefined;
        this._panel.dispose();
        while (this._disposables.length) {
            const disposable = this._disposables.pop();
            if (disposable) disposable.dispose();
        }
    }
}
