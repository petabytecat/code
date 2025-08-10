import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === CONFIGURATION ===
csv_path = "wandb_export_2025-08-08T23_12_59.957+03_00.csv"  # Replace with your file name
agents = ["DQN", "Rainbow DQN", "rainbow_dqn_sweep_config4"]  # List of agents

# "DQN", "Rainbow DQN", "rainbow_dqn_sweep_config1", "rainbow_dqn_sweep_config2", "rainbow_dqn_sweep_config3", "rainbow_dqn_sweep_config4"
metrics = ["charts/episodic_return", "charts/episodic_length"]  # Any metric(s) you want
window = 500  # Rolling average window size

agent_labels = {
    "rainbow_dqn_sweep_config4": "Rainbow DQN Sweep Config 4",
}

agent_colors = {
    "DQN": "tab:blue",
    "Rainbow DQN": "tab:red",
    "rainbow_dqn_sweep_config4": "tab:green",
}


# === LOAD & CLEAN ===
df = pd.read_csv(csv_path)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric or NaN

# === PLOTTING FUNCTION ===
def plot_metric(metric):
    plt.figure(figsize=(10, 6))

    for agent in agents:
        col = f"{agent} - {metric}"
        if col not in df.columns:
            continue

        x = df["global_step"]
        y = df[col]
        valid_mask = y.notna()

        rolling_mean = y.rolling(window=window, min_periods=1).mean()
        rolling_std = y.rolling(window=window, min_periods=1).std()

        # Identify last `tail_cut` valid points
        tail_cut = 5
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) > tail_cut:
            cutoff_idx = valid_indices[-tail_cut]
            rolling_mean[cutoff_idx:] = np.nan
            rolling_std[cutoff_idx:] = np.nan

        color = agent_colors.get(agent, None)  # Default to None if not specified
        label = agent_labels.get(agent, agent)  # fallback to agent key if no label found

        plt.plot(x, rolling_mean, label=label, color=color)
        plt.fill_between(x, rolling_mean - rolling_std, rolling_mean + rolling_std,
                        alpha=0.2, linewidth=0, color=color)

    plt.title(metric.replace("charts/", "").replace("_", " ").title())
    plt.xlabel("Global Step")
    plt.ylabel(metric.split("/")[-1].replace("_", " ").title())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === LOOP THROUGH ALL METRICS ===
for metric in metrics:
    plot_metric(metric)
