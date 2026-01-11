import * as vscode from 'vscode';
import { SearchPanel } from './searchPanel';

export function activate(context: vscode.ExtensionContext) {
    console.log('Coarch is now active');

    const searchCmd = vscode.commands.registerCommand('coarch.search', async () => {
        SearchPanel.createOrShow(context.extensionUri);
    });

    const searchSelectionCmd = vscode.commands.registerCommand('coarch.searchSelection', async () => {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            const selection = editor.document.getText(editor.selection);
            if (selection) {
                SearchPanel.createOrShow(context.extensionUri, selection);
            } else {
                vscode.window.showWarningMessage('No code selected');
            }
        }
    });

    const indexRepoCmd = vscode.commands.registerCommand('coarch.indexRepo', async () => {
        const folder = vscode.workspace.workspaceFolders?.[0];
        if (folder) {
            const result = await vscode.window.showInputBox({
                prompt: 'Enter repository path (or leave empty for current workspace)',
                value: folder.uri.fsPath,
            });
            if (result !== undefined) {
                vscode.window.withProgress({
                    location: vscode.ProgressLocation.Notification,
                    title: 'Indexing repository...',
                    cancellable: true,
                }, async (progress) => {
                    try {
                        const response = await fetch(`${getServerUrl()}/index/repo`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ path: result, name: folder.name }),
                        });
                        if (response.ok) {
                            const stats = await response.json();
                            vscode.window.showInformationMessage(
                                `Indexed ${stats.stats?.files_indexed || 0} files with ${stats.stats?.chunks_created || 0} chunks`
                            );
                        } else {
                            vscode.window.showErrorMessage('Failed to index repository');
                        }
                    } catch (error) {
                        vscode.window.showErrorMessage(`Error: ${error}`);
                    }
                });
            }
        } else {
            vscode.window.showErrorMessage('No workspace folder open');
        }
    });

    const statusCmd = vscode.commands.registerCommand('coarch.status', async () => {
        try {
            const response = await fetch(`${getServerUrl()}/index/status`);
            if (response.ok) {
                const status = await response.json();
                vscode.window.showInformationMessage(
                    `Coarch: ${status.stats?.vectors_indexed || 0} vectors indexed`,
                    'Show Details'
                ).then(selection => {
                    if (selection === 'Show Details') {
                        vscode.window.showInformationMessage(JSON.stringify(status, null, 2));
                    }
                });
            } else {
                vscode.window.showErrorMessage('Coarch server not responding');
            }
        } catch (error) {
            vscode.window.showErrorMessage(`Error connecting to Coarch: ${error}`);
        }
    });

    context.subscriptions.push(searchCmd);
    context.subscriptions.push(searchSelectionCmd);
    context.subscriptions.push(indexRepoCmd);
    context.subscriptions.push(statusCmd);
}

export function deactivate() {}

function getServerUrl(): string {
    return vscode.workspace.getConfiguration('coarch').get('serverUrl') || 'http://localhost:8000';
}
