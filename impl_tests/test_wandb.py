import wandb

wandb.login()

# 1. Start a W&B run
wandb.init(project="gpt3")

wandb.log({"acc": 60.43, "loss": 1.334})
wandb.log({"acc": 74.43, "loss": 0.834})
wandb.log({"acc": 99.43, "loss": 0.234})
wandb.config.update({"lr": 0.1, "a": 10})
# Mark the run as finished

artifact = wandb.Artifact("animals", type="dataset")
artifact.add_file("my-dataset.txt")
wandb.log_artifact(artifact)  # Creates `animals:v0`

wandb.finish()