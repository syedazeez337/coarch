" coarch.nvim - Neovim plugin for Coarch code search
" Author: syedazeez337
" License: MIT

if exists('g:coarch_loaded')
    finish
endif
let g:coarch_loaded = 1

" Configuration
let g:coarch_server_url = get(g:, 'coarch_server_url', 'http://localhost:8000')
let g:coarch_max_results = get(g:, 'coarch_max_results', 20)
let g:coarch_use_float = get(g:, 'coarch_use_float', 1)

" Commands
command! -nargs=* -complete=file CoarchSearch call coarch#search(<f-args>)
command! -nargs=? CoarchIndex call coarch#index_repo(<f-args>)
command! CoarchStatus call coarch#status()
command! CoarchServe call coarch#serve()

" Mappings
nnoremap <leader>cs :CoarchSearch 
nnoremap <leader>cg :CoarchSearch <cword><CR>
nnoremap <leader>ci :CoarchIndex 
nnoremap <leader>cc :CoarchStatus<CR>

" Autocommands
augroup coarch
    autocmd!
    " Add more autocommands as needed
augroup END

echo "Coarch loaded. Use :CoarchSearch <query> to search code"
