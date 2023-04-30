FROM bitnami/pytorch:1.2.0

WORKDIR /FHN

USER root

RUN apt-get update -y
RUN apt-get upgrade -y

RUN apt-get install python-pip -y
RUN apt-get install vim -y

COPY . .

RUN pip install -r requirements.txt
# RUN export PYTHONPATH=$PYTHONPATH:/FHN
CMD ["/bin/bash"]