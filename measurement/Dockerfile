FROM python:3.9

WORKDIR /app
COPY . /app

# Update PIP
RUN python -m pip install --upgrade pip

# Install requirements via cache
RUN --mount=type=cache,target=/root/.cache \
    pip install -r requirements.txt

RUN pip install .

ENTRYPOINT ["python"]
#ENTRYPOINT ["sh", "./measurement/utils/download_squad_data.sh"]
# Specific Utility to Run
#CMD [""]