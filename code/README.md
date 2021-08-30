## Requirements
In order to use the model (training/predicting), you'll need python 3.7.7, numpy 1.18.5, tensorflow 2.3.0, tensorflow-addons 0.11.2
and tqdm 4.46.1.

For the pre-processing steps, you'll additionally need to install nltk 3.5, pandas 1.0.4 and stanza 1.0.1.
## General Instructions
### For pre-processing
Please run the following python code:

```[python]
import stanza
stanza.download('en')
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```
## How to run FIQG
### What you will need to provide
#### Dataset
You will need to provide 3 files `test.data.json`, `dev.data.json` and `train.data.json` which shall be arrays of
elements with the following fields:
- `base_question`: Base/original question
- `base_question_pos_tags`: Base question where each word is replaced by its POS tag,
- `base_question_entity_tags`: Base question where each word is replaced by a tag which indicates if
 they are an named entity.
- `base_question_letter_cases`: Base question where each word is replaced by "UP" if it's an uppercase
word and "LOW" if it's a lowercase one.
- `base_question_ner`: Base question where each word is replaced by its NER tag.
- `facts`: List of tokenized facts.
- `facts_entity_tags`: Same as base question.
- `facts_pos_tags`: Same as base question.
- `facts_letter_cases`: Same as base question.
- `facts_ner`: Same as base question.
- `target`: Rewrite/paraphrase target
- `is_synthetic`: If the target was created by a human or a machine. If set to `True`, this example will not
 be used for testing or evaluating.
 
 Every string shall be tokenized and lower cased. An example showing how to go about
creating this file can be found in `data_processing.data_generator.generate_repeat_q_squad_raw`.
#### GloVe
If you decide to use the default parameters, you will need a [GloVe embedding](https://nlp.stanford.edu/projects/glove/).
file. We used glove.840B.300d.txt for our experiments. Place it under `/data/glove.840B.300d.txt`.
### Data Processing Step
Next, you will need to run `model.repeat_q` in `preprocess` mode, passing in argument
the name of a directory containing the JSON files mentioned above. This will create a vocabulary file, a feature
vocabulary file and optionally an embedding matrix file for you. You may or may not also have specify `-save_data_dir path`
to save the files to a specific directory. The default will be `/data/processed/repeat_q/`. Here is an example command:
```bash
python -m model.repeat_q preprocess -ds_name=fipsq_triples
```
By default, the script will look into `/data/raw_datasets/` for a directory
with the name passed in `-ds_name`. You can also specify a full path by passing it to `-preprocess_data_dir`. Please
refer back to the indications for GloVe above if you're using pretrained embeddings.
### Training
You can now train the model using `models.repeat_q` in `training` mode. Here is command to train with
default parameter settings:

```bash
python -m model.repeat_q train -ds_name=fipsq_triples --save_model
```

Your model will be saved to `/models/trained/repeat_q/` followed by either the path you passed in `-save_directory_name`
or nothing if you didn't.

### Inference
The command for running inference with default settings:
```bash
python -m model.repeat_q translate -ds_name=fipsq_triples -checkpoint_name=*a_checkpoint_name* -prediction_file_name=fipsq_triples
```

The checkpoint name is a name of a file located in `/models/trained/repeat_q`.
The output predictions can be found in `/data/results/repeat_q`
### Other options
Please refer to the arguments' descriptions for
more information by running:
```bash
python -m model.repeat_q --help
```

