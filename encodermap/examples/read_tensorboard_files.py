import os


def subdirectories(path):
    return next(os.walk(path))[1]


runs_path = "/home/tobias/PycharmProjects/encoder_map_public/encodermap/examples/runs"

runs = [d for d in subdirectories(runs_path)
        if d[:3] == "run"]

for run in runs:
    event_paths = os.walk(os.path.join(runs_path, run, "train"))

pass
