import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = f"{ROOT_DIR}/models"
REPEAT_Q_MODEL_DIR = f"{MODELS_DIR}/repeat_q"

DATA_DIR = f"{ROOT_DIR}/data"
PROCESSED_DATA_DIR = f"{DATA_DIR}/processed"

SQUAD_DIR = f"{DATA_DIR}/squad_dataset"
SQUAD_TRAIN = f"{SQUAD_DIR}/train-v1.1.json"
SQUAD_DEV = f"{SQUAD_DIR}/dev-v1.1.json"

SQUAD_REWRITE_MTURK_DIR = f"{SQUAD_DIR}/mturk"
SQUAD_REWRITES_SYNTHETIC_JSON = f"{SQUAD_DIR}/synthetic.train.json"

RESULTS_DIR = f"{DATA_DIR}/results"
TRAINED_MODELS_DIR = f"{MODELS_DIR}/trained"
PRETRAINED_MODELS_DIR = f"{MODELS_DIR}/pre_trained"

LOGS_DIR = f"{TRAINED_MODELS_DIR}/logs"
GRADIENT_DIR = f"{LOGS_DIR}/gradient_tape"

GLOVE_PATH = f"{DATA_DIR}/glove.840B.300d.txt"

# RepeatQ related definitions
REPEAT_Q_DATA_DIR = f"{PROCESSED_DATA_DIR}/repeat_q"
REPEAT_Q_RAW_DATASETS = f"{DATA_DIR}/raw_datasets"
PAD_TOKEN = "<blank>"
UNKNOWN_TOKEN = "<unk>"
EOS_TOKEN = "<eos>"

REPEAT_Q_PREDS_OUTPUT_DIR = f"{RESULTS_DIR}/repeat_q"
REPEAT_Q_SQUAD_OUTPUT_FILEPATH = f"{REPEAT_Q_PREDS_OUTPUT_DIR}/prediction.txt"
REPEAT_Q_SQUAD_DATA_DIR = f"{REPEAT_Q_DATA_DIR}/squad"
REPEAT_Q_EMBEDDINGS_FILENAME = "embeddings.npy"
REPEAT_Q_VOCABULARY_FILENAME = "vocabulary.txt"
REPEAT_Q_FEATURE_VOCABULARY_FILENAME = "feature_vocabulary.txt"
REPEAT_Q_TRAIN_CHECKPOINTS_DIR = f"{TRAINED_MODELS_DIR}/repeat_q"
