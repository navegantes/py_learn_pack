# ml_learn

## Detecção de Objetos

Uma forma simples de detectar objetos em imagens utilizando a biblioteca [cvlib](https://github.com/arunponnusamy/cvlib).

![title](obj_detect/img/traffic2_box.jpg)

## Instalação

### Instalando de um arquivo de ambiente (`environment_obj.yml`).

Entre na pasta /obj_detect e crie o ambiente virtual com os seguintes comandos.

```Bash
$ cd obj_detect
$ conda env create -f environment_obj.yml
```

Com o arquivo `environment_obj.yml` o ambiente virtual será criado com a seguintes configuração:

```yml
name: obj_det
channels:
  - defaults
dependencies:
  - python=3.9
  - pip
  - numpy=1.21.*
  - pandas=1.3.5
  - matplotlib=3.5.0
  - tensorflow-gpu=2.6.0

  - pip:
      - opencv-python==4.5.3.56
      - cvlib
```

<!-- Caso você não tenha instalado os drivers Nvidia (CUDA ToolKit, CuDNN etc)
basta alterar o pacote tensorflow-gpu=2.6.0 para tensorflow=2.6.0 -->

Ative o ambiente virtual com o comando

```
$ conda activate obj_det
```
