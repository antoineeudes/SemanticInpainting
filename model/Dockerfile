FROM eu.gcr.io/ml-mva/dependencies:1.6
RUN mkdir /app/
COPY . /app/
WORKDIR /app/
ENV PYTHONUNBUFFERED 1
CMD ["python", "workflow.py"]