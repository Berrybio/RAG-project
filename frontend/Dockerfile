FROM nginx:alpine

# nginx.conf uses ${BACKEND_URL}. The official nginx entrypoint runs envsubst
# on /etc/nginx/templates/*.template → /etc/nginx/conf.d/.
# NGINX_ENVSUBST_FILTER restricts substitution to only $BACKEND_URL so nginx's
# own variables ($uri, $remote_addr, etc.) are left untouched.
ENV BACKEND_URL=http://localhost:8000
ENV NGINX_ENVSUBST_FILTER=BACKEND_URL
COPY nginx.conf /etc/nginx/templates/default.conf.template
COPY index.html styles.css app.js /usr/share/nginx/html/

EXPOSE 80
