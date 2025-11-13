# Your main Dockerfile becomes tiny
FROM my-ml-base:1.0
WORKDIR /app
COPY serving/ ./serving/
EXPOSE 5001
ENTRYPOINT ["python", "serving/serve.py"]