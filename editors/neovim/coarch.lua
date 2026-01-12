" coarch.lua - Coarch plugin for Neovim (Lua version)
" Author: syedazeez337
" License: MIT

local coarch = {}

coarch.config = {
    server_url = 'http://localhost:8000',
    max_results = 20,
    use_float = true,
    height = 0.4,
    width = 0.8,
}

function coarch.setup(opts)
    opts = opts or {}
    coarch.config = vim.tbl_deep_extend('force', coarch.config, opts)
end

function coarch.search(query)
    if query == nil or query == '' then
        vim.ui.input({prompt = 'Search query: '}, function(input)
            if input then
                coarch._do_search(input)
            end
        end)
    else
        coarch._do_search(query)
    end
end

function coarch.search_word()
    local word = vim.fn.expand('<cword>')
    coarch._do_search(word)
end

function coarch._do_search(query)
    local url = coarch.config.server_url .. '/search'
    local body = vim.fn.json_encode({
        query = query,
        limit = coarch.config.max_results
    })

    vim.fn.jobstart({'curl', '-s', '-X', 'POST', '-H', 'Content-Type: application/json', '-d', body, url}, {
        onstdout = function(_, data, _)
            if data and data ~= '' then
                local ok, result = pcall(vim.fn.json_decode, table.concat(data, '\n'))
                if ok and result then
                    coarch._show_results(result, query)
                end
            end
        end,
        onstderr = function(_, err, _)
            vim.notify('Coarch error: ' .. table.concat(err, '\n'), vim.log.levels.ERROR)
        end
    })
end

function coarch._show_results(results, query)
    if #results == 0 then
        vim.notify('No results found for: ' .. query, vim.log.levels.WARNING)
        return
    end

    local lines = {}
    local items = {}

    for i, r in ipairs(results) do
        table.insert(lines, string.format('[%d] %s:%s (%s)', i, r.file_path, r.lines, r.language))
        table.insert(items, {
            filename = r.file_path,
            lnum = tonumber(string.match(r.lines, '%d+')) or 1,
            text = r.code:sub(1, 100),
            score = r.score
        })
    end

    vim.ui.select(lines, {
        prompt = 'Search results for: ' .. query,
        format_item = function(item)
            return item
        end,
    }, function(choice)
        if choice then
            for i, line in ipairs(lines) do
                if line == choice then
                    local item = items[i]
                    vim.cmd('edit +' .. item.lnum .. ' ' .. vim.fn.fnameescape(item.filename))
                    return
                end
            end
        end
    end)
end

function coarch.index_repo(path)
    path = path or vim.fn.expand('%:p:h')
    local name = vim.fn.input('Repository name: ', vim.fn.fnamemodify(path, ':t'))

    local url = coarch.config.server_url .. '/index/repo'
    local body = vim.fn.json_encode({
        path = path,
        name = name
    })

    vim.notify('Indexing repository: ' .. path, vim.log.levels.INFO)

    vim.fn.jobstart({'curl', '-s', '-X', 'POST', '-H', 'Content-Type: application/json', '-d', body, url}, {
        onstdout = function(_, data, _)
            if data and data ~= '' then
                vim.notify('Indexing complete: ' .. table.concat(data, '\n'), vim.log.levels.INFO)
            end
        end,
        onstderr = function(_, err, _)
            vim.notify('Indexing error: ' .. table.concat(err, '\n'), vim.log.levels.ERROR)
        end
    })
end

function coarch.status()
    local url = coarch.config.server_url .. '/index/status'

    vim.fn.jobstart({'curl', '-s', url}, {
        onstdout = function(_, data, _)
            if data and data ~= '' then
                local ok, result = pcall(vim.fn.json_decode, table.concat(data, '\n'))
                if ok and result then
                    local msg = string.format('Coarch: %d vectors indexed', result.stats and result.stats.vectors_indexed or 0)
                    vim.notify(msg, vim.log.levels.INFO)
                end
            end
        end,
        onstderr = function(_, err, _)
            vim.notify('Status error: ' .. table.concat(err, '\n'), vim.log.levels.ERROR)
        end
    })
end

function coarch.serve()
    vim.notify('Starting Coarch server...', vim.log.levels.INFO)
    vim.fn.jobstart({'python', '-m', 'backend.server'}, {
        onstdout = function(_, data, _)
            if data then
                vim.notify(table.concat(data, '\n'), vim.log.levels.INFO)
            end
        end,
        onstderr = function(_, data, _)
            if data then
                vim.notify(table.concat(data, '\n'), vim.log.levels.ERROR)
            end
        end,
        onexit = function(_, code, _)
            if code == 0 then
                vim.notify('Coarch server stopped', vim.log.levels.INFO)
            else
                vim.notify('Coarch server exited with code: ' .. code, vim.log.levels.ERROR)
            end
        end
    })
end

return coarch
