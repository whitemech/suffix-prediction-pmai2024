FROM nvcr.io/nvidia/pytorch:21.04-py3

# Set non-interactive mode for apt-get when asking for user input
ARG DEBIAN_FRONTEND=noninteractive

# Set the timezone (necessary for installing tzdata)
ENV TZ=Europe/Rome
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y \
    build-essential \
    software-properties-common \
    bash-completion \
    htop \
    curl \
    wget \
    vim \
    nano \
    graphviz \
    git \
    flex

# Install MONA
RUN wget http://www.brics.dk/mona/download/mona-1.4-18.tar.gz
RUN tar -xvzf mona-1.4-18.tar.gz
RUN cd mona-1.4 && ./configure && make && make install
# Add MONA lib to $LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/lib

# Start with bash
ENTRYPOINT ["/bin/bash"]
