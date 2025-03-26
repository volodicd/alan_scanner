FROM ubuntu:latest
LABEL authors="volodic"

ENTRYPOINT ["top", "-b"]