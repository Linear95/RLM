TRAIN_DATA_PATH="/path/to/train/data/dir/masked_train_data.json"
TEST_DATA_PATH="../test/test_data.json"
SAVE_DIR="/saved_models/style_classifer/"

python train_style_classifier.py\
    --train_data_path ${TRAIN_DATA_PATH}\
    --test_data_path ${TEST_DATA_PATH}\
    --save_dir ${SAVE_DIR}\
    --batch_size 64
