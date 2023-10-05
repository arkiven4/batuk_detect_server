FROM python:3.8.18-alpine3.17
WORKDIR /srv
RUN pip install --upgrade pip
RUN pip install torch torchaudio torchvision librosa==0.8.0 opencv-python tqdm nlpaug cmapy audiomentations timm==0.4.5 wget cmake==3.18.4 seaborn numpy flask
COPY . /srv
ENV FLASK_APP=app
CMD ["python","flask_app.py"]
