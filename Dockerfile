FROM python:3.5
RUN mkdir -p /usr/src/app
RUN mkdir -p /usr/src/app/log
WORKDIR /usr/src/app
COPY requirements.txt /usr/src/app/
RUN pip install --no-cache-dir -r requirements.txt
#ON DEBUG RUN PIP LIB FROM VOLUME
#RUN pip install -e /shared/MessageUtilities
COPY run.py /usr/src/app
COPY ./Model /usr/src/app/Model
COPY ./Runner /usr/src/app/Runner
COPY Expert.csv /usr/src/app/Expert.csv 

COPY savedmodels /usr/src/app/savedmodels

COPY cmd.sh /usr/src/app/cmd.sh 
CMD ["bash","/usr/src/app/cmd.sh"]
#CMD [ "python","-u", "/usr/src/app/run.py" ]