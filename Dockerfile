FROM python:latest
COPY ./requirements.txt .
RUN apt update -y
RUN apt install -y r-base
RUN pip install -r requirements.txt