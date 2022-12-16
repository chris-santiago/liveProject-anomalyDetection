FROM python:3.7-slim

RUN mkdir src
WORKDIR /src

# python requirements
COPY requirements.txt /src
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# install nodejs for plotly and juypter lab
# nodejs install dependent on gnu
# install plotly widget for jupyter lab
RUN apt-get update
RUN apt-get -y install curl gnupg
RUN curl -sL https://deb.nodesource.com/setup_12.x | bash -
RUN apt-get install nodejs -y
RUN jupyter labextension install jupyterlab-plotly

CMD ["jupyter", "lab", "--no-browser", "--allow-root", "--ip=0.0.0.0", "--port=8888"]