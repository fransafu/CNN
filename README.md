# Tarea 1

# Declaración

Este proyecto se encuentra basado en el proyecto `convnet2` del usuario **jmsaavedrar**, puede visitar el proyecto base en el siguiente enlace: [click aquí](https://github.com/jmsaavedrar/convnet2).

# Install dependencies
* pip install -r requirements.txt

# Como entrenar

Se debe especificar el archivo de configuración definiendo la ruta obsoluta en el parametro `-config` y definir el modelo a utilizar `-model`. Por ejemplo:

* python train.py -config '/path/to/folder/configs/config_model_v2.yaml' -model resnet

El Script train almacenará los resultado en la carpeta `experiments`.
