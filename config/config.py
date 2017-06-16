import os

is_local = "LOCAL" in os.environ

if is_local:
    config = {
        "output_dir": ".",
        "data_root_dir": "/home/faruk/Desktop/experiment_root",
        "inception-top": {
            "batch_size": 2,
            "epoch_n": 2,
            "data_n": 5
        }
    }
else:
    config = {
        "output_dir": "/output",
        "data_root_dir": "/input",
        "inception-top": {
            "batch_size": 1000,
            "epoch_n": 500,
            "data_n": 10000
        }
    }