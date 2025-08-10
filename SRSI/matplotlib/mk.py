import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === CONFIGURATION ===
csv_files = {
    "DQN": "dqn_run.csv",
    "Rainbow DQN": "rainbow_dqn_run.csv",
    "rainbow_dqn_sweep_config4": "rainbow_dqn_sweep_config4_run.csv"
}

agents = ["DQN", "Rainbow DQN", "rainbow_dqn_sweep_config4"]
metrics = ["charts/episodic_return", "charts/episodic_length"]
window = 500

agent_labels = {
    "rainbow_dqn_sweep_config4": "Rainbow DQN Sweep Config 4",
}

agent_colors = {
    "DQN": "tab:blue",
    "Rainbow DQN": "tab:red",
    "rainbow_dqn_sweep_config4": "tab:green",
}

# === LOAD & COMBINE DATA ===
def load_and_combine_data():
    combined_df = pd.DataFrame()
    
    for agent in agents:
        try:
            df = pd.read_csv(csv_files[agent])
            
            # Convert columns to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Rename columns to match your original format
            for metric in metrics:
                if metric in df.columns:
                    df[f"{agent} - {metric}"] = df[metric]
            
            # Only keep global_step and the renamed metric columns
            keep_cols = ['global_step'] + [f"{agent} - {metric}" for metric in metrics]
            df = df[keep_cols]
            
            # Merge with combined_df
            if combined_df.empty:
                combined_df = df
            else:
                combined_df = pd.merge(combined_df, df, on='global_step', how='outer')
            
        except Exception as e:
            print(f"Error loading {agent}: {e}")
    
    return combined_df

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

        color = agent_colors.get(agent, None)
        label = agent_labels.get(agent, agent)

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

# === MAIN EXECUTION ===
df = load_and_combine_data()
for metric in metrics:
    plot_metric(metric)