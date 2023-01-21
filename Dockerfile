FROM ubuntu:20.04

RUN apt-get update && apt-get install -y curl
RUN curl -sL https://deb.nodesource.com/setup_14.x | bash
RUN apt-get install nodejs
RUN node -v
RUN npm -v

ADD ./third_party/pokemon-showdown /showdown/pokemon-showdown

EXPOSE 8000

WORKDIR /showdown/pokemon-showdown

RUN mkdir -p /showdown/pokemon-showdown/logs/repl && \
    cp /showdown/pokemon-showdown/config/config-example.js /showdown/pokemon-showdown/config/config.js && \
    npm install --global n && \
    n latest

#CMD [ "node", "pokemon-showdown", "start", "--no-security" ]
