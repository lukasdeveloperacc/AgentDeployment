# Frontend Dockerfile for Static Web Files
# Nginx 경량 이미지 사용
FROM nginx:alpine

# Nginx 설정 파일 복사 (커스텀 설정)
COPY nginx.conf /etc/nginx/nginx.conf

# 정적 파일 복사
COPY index.html /usr/share/nginx/html/
COPY app.js /usr/share/nginx/html/
COPY style.css /usr/share/nginx/html/

# 포트 노출
EXPOSE 80

# Nginx 실행 (기본 CMD 사용)
