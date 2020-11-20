OBS: Por problemas na replicação do ENV de forma automática, usando o famoso requirementes, sugiro que façam a instalação manual, seguindo os seguintes passos:

1) Criem um env com um nome de seu interesse <env_name> com python3.6 e, para isso, use o comando:  conda create -n <env_name> python=3.6

2) EM seguida, entre no <env_name> como comando: conda activate <env_name>

3) Instale nessa ordem as seguintes bibliotecas:

	conda install -c conda-forge dlib
	pip install face-recognition==1.2.3
	pip install opencv-python


Esta atividade será construir um script .py para capturar frames da webcam ou de um arquivo de vídeo (se preferir) e
realizar um reconhecimento facial de sua própria face ou de alguém de escolha que possa estar presente no vídeo ou à frente da webcam! Vamos construir durante a aula!


