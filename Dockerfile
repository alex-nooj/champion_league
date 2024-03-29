FROM node as showdown

ADD ./third_party/pokemon-showdown /showdown/pokemon-showdown

EXPOSE 8000

WORKDIR /showdown/pokemon-showdown

RUN mkdir -p /showdown/pokemon-showdown/logs/repl && \
    npm install --global n && \
    n latest

CMD [ "node", "pokemon-showdown", "start", "--no-security" ]
