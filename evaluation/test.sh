TRAIN_DATA_PATH="/path/to/train/data/dir/masked_train_data.json"
TEST_DATA_PATH="../test/test_data.json"
SAVE_DIR="/saved_models/style_classifer"
MODEL_PATH="${SAVE_DIR}/new_roberta_yelp_classifier_epoch_4"

python style_classification.py\
    --evaluate\
    --train_data_path ${TRAIN_DATA_PATH}\
    --test_data_path ${TEST_DATA_PATH}\
    --model_path ${MODEL_PATH}\
    --tokenizer_path roberta-base\
    --batch_size 64
