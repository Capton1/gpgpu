FROM ubuntu:20.04

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" TZ="Europe/Paris" apt-get install -y vim cmake libpng-dev libtbb-dev libopencv-dev
RUN apt-get install build-essential -y


RUN 

RUN mkdir /workdir

RUN export uid=1000 gid=1000 && \
    mkdir -p /home/developer && \
    echo "developer:x:${uid}:${gid}:Developer,,,:/home/developer:/bin/bash" >> /etc/passwd && \
    echo "developer:x:${uid}:" >> /etc/group && \
    chown ${uid}:${gid} -R /home/developer && \
    chown ${uid}:${gid} -R /workdir

USER developer

WORKDIR /workdir

CMD bash