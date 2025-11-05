#-----------------BUILD STAGE----------------------------
# TODO: Use an official Python runtime as a parent image
FROM python:3.11-slim AS builder
# TODO: Set the working directory in the container
WORKDIR /app
# TODO: Copy the dependencies file to the working directory
COPY requirements.txt . 
# TODO: Install any needed packages specified in requirements.txt
#run pip install --no-cache-dir -r ./requirements.txt
RUN pip install -r ./requirements.txt --target /install
#----------- FINAL RUNTIME ENVIRONMENT-----------------
FROM python:3.11-slim

#copy installed dependencies , binaries from builder
COPY --from=builder /install /usr/local/lib/python3.11/site-packages
# TODO: Copy the rest of the application's code
COPY . . 
#expose port
EXPOSE 8080
# Set the ENTRYPOINT run the serving scirpt when the container starts 
ENTRYPOINT ["python" , "serving/serve.py"]
# TODO: Command to run the application
CMD ["python","serving/serve.py"]
