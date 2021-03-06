from defs import REPEAT_Q_SQUAD_DATA_DIR, REPEAT_Q_VOCABULARY_FILENAME, REPEAT_Q_FEATURE_VOCABULARY_FILENAME


class ModelConfiguration:

    def __init__(self,
                 nb_epochs=20,
                 dropout_rate=0.5,
                 attention_dropout_rate=0.3,
                 recurrent_dropout=0.0,
                 fact_encoder_hidden_size=256,
                 max_generated_question_length=50,
                 question_attention_function="additive",
                 facts_attention_function="additive",
                 attention_depth=512,
                 embedding_size=300,
                 decoder_hidden_size=256,
                 decoder_readout_size=128,
                 batch_size=32,
                 data_dir=REPEAT_Q_SQUAD_DATA_DIR,
                 restore_checkpoint=False,
                 model_checkpoint_path=None,
                 synth_supervised_epochs=2,
                 org_supervised_epochs=18,
                 dev_step_size=-1,
                 learning_rate=None,
                 saving_model=False,
                 training_beam_search_size=5,
                 use_ner_features=True,
                 use_pos_features=True,
                 mixed_data=False,
                 reduced_ner_indicators=False,
                 use_question_encodings=True,
                 use_glove_embeddings=True):
        super(ModelConfiguration, self).__init__()
        self.recurrent_dropout = recurrent_dropout
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.fact_encoder_hidden_size = fact_encoder_hidden_size
        self.embeddings_pretrained = use_glove_embeddings
        self.embedding_size = embedding_size
        self.data_dir = data_dir
        self.question_attention = question_attention_function
        self.facts_attention = facts_attention_function
        self.attention_depth = attention_depth
        self.decoder_hidden_size = decoder_hidden_size
        self.max_generated_question_length = max_generated_question_length
        self.decoder_readout_size = decoder_readout_size
        self.batch_size = batch_size
        self.restore_supervised_checkpoint = restore_checkpoint
        self.supervised_model_checkpoint_path = model_checkpoint_path
        self.synth_supervised_epochs = synth_supervised_epochs
        self.org_supervised_epochs = org_supervised_epochs
        self.epochs = nb_epochs
        self.dev_step_size = dev_step_size
        self.learning_rate = learning_rate
        self.saving_model = saving_model
        self.training_beam_search_size = training_beam_search_size
        self.use_ner_features = use_ner_features
        self.use_pos_features = use_pos_features
        self.mixed_data = mixed_data
        self.reduced_ner_indicators = reduced_ner_indicators
        self.use_question_encodings = use_question_encodings
        self.save_directory_name = None

    @staticmethod
    def new() -> 'ModelConfiguration':
        config = ModelConfiguration()
        return config

    @property
    def vocabulary_path(self):
        return f"{self.data_dir}/{REPEAT_Q_VOCABULARY_FILENAME}"

    @property
    def feature_vocabulary_path(self):
        return f"{self.data_dir}/{REPEAT_Q_FEATURE_VOCABULARY_FILENAME}"

    def with_data_dir(self, data_dir):
        self.data_dir = data_dir
        return self

    def with_glove_embeddings(self, use_glove_embeddings):
        self.embeddings_pretrained = use_glove_embeddings
        return self

    def with_question_encodings(self, use_question_encodings):
        """
        :param use_question_encodings: If set to False, the base question encoder will be skipped and the question
        embeddings will be directly used instead (for instance through
        """
        self.use_question_encodings = use_question_encodings
        return self

    def with_reduced_ner_indicators(self, reduce_indicators):
        """
        :param reduce_indicators: Whether to use the reduced set of named entity indicators or the normal one. Normal
        one means we will differentiate between answer named entities and base question entities.
        """
        self.reduced_ner_indicators = reduce_indicators
        return self

    def with_mixed_data(self, mixed_data):
        self.mixed_data = mixed_data
        return self

    def with_ner_features(self, use_ner_features):
        self.use_ner_features = use_ner_features
        return self

    def with_pos_features(self, use_pos_features):
        self.use_pos_features = use_pos_features
        return self

    def with_attention_dropout(self, attention_dropout):
        self.attention_dropout_rate = attention_dropout
        return self

    def with_dropout_rate(self, dropout_rate):
        self.dropout_rate = dropout_rate
        return self

    def with_recurrent_dropout(self, recurrent_dropout):
        self.recurrent_dropout = recurrent_dropout
        return self

    def with_batch_size(self, batch_size):
        self.batch_size = batch_size
        return self

    def with_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        return self

    def with_restore_supervised_checkpoint(self):
        self.restore_supervised_checkpoint = True
        return self

    def with_supervised_model_checkpoint_path(self, ckpt_path):
        self.supervised_model_checkpoint_path = ckpt_path
        return self

    def with_saving_model(self, save_model: bool):
        self.saving_model = save_model
        return self

    def with_save_directory_name(self, save_directory_name: str):
        self.save_directory_name = save_directory_name
        return self

    def with_org_supervised_epochs(self, nb_epochs):
        self.org_supervised_epochs = nb_epochs
        return self

    def with_synth_supervised_epochs(self, nb_epochs):
        self.synth_supervised_epochs = nb_epochs
        return self

    def with_dev_step_size(self, dev_step_size):
        self.dev_step_size = dev_step_size
        return self

    def __str__(self):
        str_builder = ""
        for param_name, default_value in self.__dict__.items():
            str_builder += f"{param_name}: {default_value}\n"
        return str_builder
