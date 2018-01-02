FROM python:3.5
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
COPY requirements.txt /usr/src/app/
RUN pip install --no-cache-dir -r requirements.txt
COPY run.py /usr/src/app
COPY __init__.py /usr/src/app
COPY ./Model /usr/src/app/Model
COPY ./Runner /usr/src/app/Runner
COPY cmd.sh /usr/src/app/cmd.sh 
CMD ["bash","/usr/src/app/cmd.sh"]