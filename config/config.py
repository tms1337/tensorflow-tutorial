import os

is_local = "LOCAL" in os.environ

if is_local:
    config = {
        "output_dir": ".",
        "data_root_dir": "/home/faruk/Desktop/experiment_root",
        "inception-top": {
            "batch_size": 2,
            "epoch_n": 5,
            "data_n": 10,
            "repeat-n": 1
        }
    }
else:
    config = {
        "output_dir": "/output",
        "data_root_dir": "/input",
        "inception-top": {
            "batch_size": 500,
            "epoch_n": 10,
            "data_n": 5000,
            "repeat-n": 1
        }
    }