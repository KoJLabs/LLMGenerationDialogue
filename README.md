
## Getting Started
To run the experiments included in this study, follow these setup instructions.

### Prerequisites


```
poetry shell
poetry install
pip3 install torch torchvision torchaudio pip3 install -q -U git+https://github.com/huggingface/peft.git
```

### Installation
Clone the repository to your local machine:

```
git clone git@github.com:KoJLabs/LLMGenerationDialogue.git
cd LLMGenerationDialogue
```


### Running the Training Script
To start the training process, use the following command:

```
poetry run python -m train --config_path train/config.yaml
```

### Running the Inference Script
To start the inference process, use the following command:

```
poetry run python -m inference --config_path inference/config.yaml
```



## Paper

```
이주환, 허탁성, 김지수, 정민수, 이경욱, 김경선. "Large Language Model을 활용한 키워드 기반 대화 생성 (Keyword Based Conversation Generation using Large Language Model)." 한국정보과학회 언어공학연구회:학술대회논문집(한글 및 한국어 정보처리), 한국정보과학회언어공학연구회 2023년도 제35회 한글 및 한국어 정보처리 학술대회, 2023, pp. 19-24.
```