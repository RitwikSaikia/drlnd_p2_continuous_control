import numpy as np

YELLOW = '#F8A32A'
RED = "#F33C68"
BLUE = "#1C93D6"
BLACK = "#555555"


def plot_scores(scores, ax=None, window_size=100, solved_score=None):
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    n_episodes = len(scores)
    window_size = min(window_size, n_episodes)

    x_data = np.asarray(range(n_episodes))
    ax.plot(x_data, scores, color=BLUE, alpha=0.3, label='Actual Score')

    scores_moving_average = np.convolve(scores, np.ones((window_size,)) / window_size, mode='valid')
    x_data = x_data[n_episodes - len(scores_moving_average):]
    ax.plot(x_data, scores_moving_average, color=BLUE, alpha=1.0, label='Avg(%d) Score' % window_size)

    if solved_score is not None:
        solved_index = np.where(scores_moving_average >= solved_score)[0]
        if len(solved_index) > 0:
            solved_index = solved_index[0]
            solved_value = scores_moving_average[solved_index]
            solved_index += window_size
            ax.plot(solved_index, solved_value, 'go-', markersize=12, color=RED,
                    label='Solved Avg Score: %.2f at %d' % (solved_value, solved_index))

    best_index = np.argmax(scores_moving_average)
    best_value = scores_moving_average[best_index]
    best_index += window_size
    ax.plot(best_index, best_value, 'go-', markersize=12, color=YELLOW,
            label='Max Avg Score: %.2f at %d' % (best_value, best_index))

    ax.set_xlabel("# Episodes", color=BLACK)
    ax.set_ylabel("Score", color=BLACK)

    ax.set_ylim([0, np.max(scores) * 1.01])
    ax.set_xlim([0, len(scores) * 1.01])

    ax.spines['bottom'].set_color(BLACK)
    ax.spines['top'].set_color(BLACK)
    ax.spines['left'].set_color(BLACK)
    ax.spines['right'].set_color(BLACK)
    ax.xaxis.label.set_color(BLACK)
    ax.yaxis.label.set_color(BLACK)
    ax.tick_params(axis='x', colors=BLACK)
    ax.tick_params(axis='y', colors=BLACK)

    ax.legend()

    return ax


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    scores = np.load("scores-final.npy")
    plot_scores(scores, solved_score=30)
    plt.savefig("plot.png")
    plt.show()
