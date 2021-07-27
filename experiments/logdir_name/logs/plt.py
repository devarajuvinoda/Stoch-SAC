import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def plot_learning_curve(x, scores, figure_file):
    moving_avg = np.zeros(len(scores))

    for i in range(len(moving_avg)):
        moving_avg[i] = np.mean(scores[max(0, i-100) : (i+1)])

    plt.plot(x, moving_avg)
    plt.title('score v/s #episodes')
    plt.savefig(figure_file)
    
if __name__=="__main__":
    df = pd.read_csv('log.csv')    
    plot_learning_curve(df["steps"], df["return"], "moving_avg.png")