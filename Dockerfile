##set the python image version in docker
FROM python:3.13.3-slim

##set/create the working directory inside docker image
WORKDIR /app

##copy all dependencies from local directory to docker image
COPY . /app

##install all dependencies from requirements.txt file
RUN pip install -r requirements.txt

##expose the local app on 5000 port
EXPOSE 5000

##command to run fast api
CMD [ "python3", "app.py" ]
