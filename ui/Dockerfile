FROM python:3.9

WORKDIR /app
COPY . /app

# Update PIP & install requirements
RUN python -m pip install --upgrade pip

# Switch to the new user

RUN --mount=type=cache,target=/root/.cache \
    pip install -r requirements.txt

EXPOSE 8080

ENV FLASK_APP=auto_app.py
ENV SECRET_KEY=not-so-secret
ENV LOG_LEVEL=info
ENV GUNICORN_WORKERS=1
ENV SEND_FILE_MAX_AGE_DEFAULT=0

CMD ["python","-m", "flask", "run", "--host", "0.0.0.0", "--port", "8080"]
