DATA_DIR="/home/pengyu/workspace/style_transfer_datasets1/yelp"
TEST_DATA_PATH="../test/test_data.json"
MODEL_DIR="/home/pengyu/workspace/saved_models/RLM_results/model"
MODEL_PATH="${MODEL_DIR}/model_step_2_17000"
SAVE_DIR="/home/pengyu/workspace/saved_models/RLM_results"
OUTPUT_DIR="/home/pengyu/workspace/saved_models/RLM_results"

python main.py\
       --evaluate\
       --test_data_path ${TEST_DATA_PATH}\
       --markers_path ${DATA_DIR}/markers.json\
       --rarewords_path ${DATA_DIR}/rare_words.json\
       --model_path ${MODEL_PATH}\
       --tokenizer_path bert-base-uncased\
       --output_dir ${OUTPUT_DIR}
