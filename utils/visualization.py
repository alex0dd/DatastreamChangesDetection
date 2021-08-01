import matplotlib.pyplot as plt

def get_plot_color(n_samples, sample_idx, max_intensity=0.8):
    return (max_intensity / n_samples) * sample_idx
    
def print_changes_time(df_indices, changes):
    print("Changes in stream distribution occurred at:")
    for change_idx in changes:
        print("\t{}".format(df_indices[change_idx]))

def plot_total_df(df, title, changes=None, max_intensity=0.6, print_changes=True):
    fig = plt.figure(figsize=(40, 10))
    ax = fig.add_subplot(111)
    df.plot(ax=ax)
    df_indices = df.index
    if changes is not None:
        if print_changes:
            print_changes_time(df_indices, changes)
        if len(changes) > 1:
            for i in range(len(changes) - 1):
                color = (1, 0, 0, get_plot_color(len(changes) + 1, i + 1, max_intensity=max_intensity))
                start_idx = df_indices[changes[i]]
                end_idx = df_indices[changes[i + 1]]
                ax.axvspan(start_idx, end_idx, color=color)
        # last change goes from last change index until the end
        start_idx = df_indices[changes[-1]]
        end_idx = df_indices[-1]
        color = (1, 0, 0, get_plot_color(len(changes) + 1, len(changes) + 1, max_intensity=max_intensity))
        ax.axvspan(start_idx, end_idx, color=color)
                
    plt.title(title)
    plt.show()
