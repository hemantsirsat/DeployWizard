services:
  {{ service_name }}:
    build: .
    ports:
      - "{{ port }}:8000"
    environment:
      - PORT=8000
      - HOST=0.0.0.0
    volumes:
      - ./app:/app
    restart: unless-stopped
