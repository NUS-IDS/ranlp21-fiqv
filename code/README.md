This code was developed in Python 3.7.7


## General Instructions
Please run the following python code:

```[python]
import stanza
stanza.download('en') 
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```
## How to run QPGen
### What you will need to provide
You will need to provide a JSON file containing your whole dataset (train+dev+test) with the following schema:
```javascript
{
    "facts": List[String],
    "base_question": String,
    "target_question": String
}
```
Every string should be tokenized and lower cased. An example showing how to go about 
creating this file can be found in `data_processing.data_generator.generate_repeat_q_squad_raw`.
### Data Processing Step
Next, you will need to run `models.repeat_q` in `preprocessing` mode, passing in argument
the path to the JSON file mentioned above. This will create a vocabulary file and optionally
an embedding matrix file for you.
### Training
You can now train the model using `models.repeat_q` in `training` mode. Please refer to the arguments' descriptions for
more information by running:
```bash
python -m models.repeat_q --help
```
