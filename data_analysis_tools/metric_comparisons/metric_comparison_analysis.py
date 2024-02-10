import pandas as pd
import matplotlib.pyplot as plt
import os

input_dir1 = "data/outputs/2024-02-08_18-25-51/metrics"
input_dir2 = "data/outputs/2024-02-08_18-31-42/metrics"
output_dir = "/home/azure_reflection/aisc/Steering-LLMs/data_analysis_tools/metric_comparisons/"

files1 = [f for f in os.listdir(input_dir1) if f.endswith('.csv')]
files2 = [f for f in os.listdir(input_dir2) if f.endswith('.csv')]

for file in set(files1) & set(files2):

    data1 = pd.read_csv(os.path.join(input_dir1, file))
    data2 = pd.read_csv(os.path.join(input_dir2, file))

    plt.plot(data1['Layer'], data1['Accuracy'], color='blue', label='Full stop ending.')
    plt.plot(data2['Layer'], data2['Accuracy'], color='red', label='Because ending')

    plt.xlabel('Layer')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs Layer for {file}')
    plt.legend()

    plt.savefig(output_dir + f'{os.path.splitext(file)[0]}_plot.png')

    # /home/azure_reflection/aisc/Steering-LLMs/data_analysis_tools/metric_comparisons

    # Clear the current figure for the next plot
    plt.clf()