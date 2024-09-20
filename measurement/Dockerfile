FROM python:3.9

WORKDIR /app
COPY . /app

# Update PIP
RUN python -m pip install --upgrade pip

# Install library and requirements
RUN --mount=type=cache,target=/root/.cache \
    pip install .

ENTRYPOINT ["python"]

# Specific Utility to Run
CMD []
