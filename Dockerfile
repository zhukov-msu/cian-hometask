FROM continuumio/miniconda3
MAINTAINER "Alex Zhukov"

RUN apt-get update && apt-get install -y libgtk2.0-dev && \
    rm -rf /var/lib/apt/lists/* && \
    /opt/conda/bin/conda install jupyter -y && \
    /opt/conda/bin/conda install -y numpy pandas scikit-learn matplotlib seaborn pyyaml h5py flask
    
RUN pip install tqdm==4.43.0 pandas_profiling folium geopy python-Levenshtein fuzzywuzzy xgboost pymorphy2==0.8 spacy==2.1.9 psycopg2-binary gensim

# Jupyter and Tensorboard ports
EXPOSE 8888 6006
EXPOSE 80 80

COPY . /cian_app
WORKDIR /cian_app
RUN git clone -b v2.1 https://github.com/buriy/spacy-ru.git
RUN tar xvf ./models/models.tgz -C ./models

ENTRYPOINT ["/opt/conda/bin/python", "app.py"]
