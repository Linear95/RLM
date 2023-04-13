DATA_DIR=/home/pengyu/workspace/style_transfer_datasets1/yelp
SAVE_DIR=/home/pengyu/workspace/saved_models/RLM_results1

python main.py\
       --train_data_path ${DATA_DIR}/masked_train_data.json\
       --markers_path ${DATA_DIR}/markers.json\
       --rarewords_path ${DATA_DIR}/rare_words.json\
       --model_path bert-base-uncased\
       --tokenizer_path bert-base-uncased\
       --save_dir ${SAVE_DIR}
