FROM python:3.8.18-alpine3.17
WORKDIR /srv
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . /srv
ENV FLASK_APP=app
CMD ["python","flask_app.py"]