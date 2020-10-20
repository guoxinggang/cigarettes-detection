mkdir -p ~/.streamlit

echo "[general]
email = gxg@bupt.edu.cn
" > ~/.streamlit/credentials.toml

echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
