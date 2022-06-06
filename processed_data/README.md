The processed data files for FIRS after feature extraction (in a format suitable for running our code)
This needs to be further processed as follows for obtaining the vocabulary and other intermediate files for running the training/test

### UPDATE to preprocessing [June 6,2022]


Place the three data files (names expected to be train.data.json test.data.json dev.data.json) under [data-dir/dataset_name_dir]=INPDIR
cd to code directory and run

python -m model.repeat_q preprocess -preprocess_data_dir=INPDIR -save_data OUTDIR -pretrained_embeddings_path=/path/to/glove.840B.300d.txt -voc_size=20000 -ds_name=dataset_name

****Note the file and directory name conventions, can change voc_size to the desired vocabulary size




