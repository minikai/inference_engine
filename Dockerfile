FROM python:3
RUN apt-get update -y
COPY \code\ /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN mkdir /inference_engine
ENTRYPOINT ["python"]
CMD ["inference.py"]
