import wandb
api = wandb.Api()

# run is specified by <entity>/<project>/<run id>
run = api.run("david-zhangblade-kaust/cleanRL/runs/565zhsoj")

# save the metrics for the run to a csv file
metrics_dataframe = run.history()
metrics_dataframe.to_csv("test.csv")
