import os

TIME_STEPS = 28
FILTER_SIZE = 28
KERNEL_SIZE = 3
NUM_TIME_SERIES = 502
NUM_FEATURES = 14
MODEL_PATH = 'models/'
MODEL_NAME = "dropout_layers_0.4_0.4" + '_' + str(TIME_STEPS) + '_' + str(NUM_TIME_SERIES) + '_' + str(NUM_FEATURES)
OUTPUT_PATH = MODEL_PATH+MODEL_NAME

if not os.path.exists(OUTPUT_PATH): os.makedirs(OUTPUT_PATH)

params = {
    "batch_size": 10,  # 20<16<10, 25 was a bust
    "epochs": 300,
    "lr": 0.00010000,
    "time_steps": 60
}

BATCH_SIZE = params["batch_size"]
