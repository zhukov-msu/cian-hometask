FROM continuumio/anaconda3
MAINTAINER "Alex Zhukov"

RUN apt-get update && apt-get install -y libgtk2.0-dev && \
    rm -rf /var/lib/apt/lists/* 
#   && \
#    /opt/conda/bin/conda install jupyter -y && \
#    /opt/conda/bin/conda install numpy pandas scikit-learn matplotlib seaborn pyyaml h5py keras -y && \
#    /opt/conda/bin/conda upgrade dask && \
    
RUN pip install tqdm pandas_profiling folium geopy python-Levenshtein fuzzywuzzy xgboost pymorphy2==0.8 spacy==2.1.9 psycopg2-binary gensim

# COPY jupyter_notebook_config.py /home/ubuntu/.jupyter/
# COPY run_jupyter.sh /

# Jupyter and Tensorboard ports
EXPOSE 8888 6006
EXPOSE 80

COPY . /cian_app
WORKDIR /cian_app
RUN git clone -b v2.1 https://github.com/buriy/spacy-ru.git
RUN tar xvf ./models/models.tgz -C ./models

ENTRYPOINT ['/opt/conda/bin/python' 'app.py']

# RUN nohup jupyter notebook --ip=0.0.0.0 --port=8080  --allow-root > log_jupyter_docker.txt &