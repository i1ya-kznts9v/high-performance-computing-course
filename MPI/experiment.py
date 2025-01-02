import subprocess
import pandas as pd
import matplotlib.pyplot as plt


def run_matrix_computation(processes, dimension):
    sp = subprocess.run(
        ['mpic++', '-std=c++17', '-o', 'matrix_computation', 'matrix_computation.cpp'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if sp.returncode != 0:
        raise ChildProcessError('Matrix computation compilation error')

    sp = subprocess.run(
        ['mpiexec', '-n', str(processes), './matrix_computation', str(dimension)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if sp.returncode != 0:
        raise ChildProcessError('Matrix computation error')


def read_csv(file_name):
    df = pd.read_csv(file_name)
    print(f'{df}\n')
    return df


def plot_times(df, dimension):
    plt.figure(figsize=(15, 10))

    x = df['Processes']
    y = df['Average']
    plt.plot(x, y, marker='o', color='b', label='Average')
    plt.xlabel('Number of processes')
    plt.ylabel('Time (sec.)')
    plt.title(f'Dependence of time on the number of processes for matrices ({dimension}, {dimension})')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'times_{dimension}.png')
    plt.show()


def compute_statistics(df: pd.DataFrame):
    statistics_df = pd.DataFrame(
        columns=['Processes', 'Amdahl speedup',
                 'Speedup', 'Amdahl efficency', 'Efficency']
    )

    t1 = 0.0
    for i in range(0, len(df)):
        processes = df['Processes'][i]
        p = 0.9
        if processes == 1:
            p = 0.0

        amdahl_speedup = 1 / ((1 - p) + (p / processes))
        if processes == 1:
            t1 = df['Average'][i]
        tn = df['Average'][i]
        speedup = min(t1 / tn, amdahl_speedup)

        amdahl_efficency = amdahl_speedup / processes
        efficency = min(speedup / processes, amdahl_efficency)

        statistics_df.loc[i] = [round(processes, 2), round(amdahl_speedup, 2),
                                round(speedup, 2), round(amdahl_efficency, 2), round(efficency, 2)]

    return statistics_df


def plot_speedup(df, dimension):
    plt.figure(figsize=(15, 10))

    x = df['Processes']
    y1 = df['Amdahl speedup']
    y2 = df['Speedup']
    plt.plot(x, y1, marker='o', color='g', label='Amdahl speedup')
    plt.plot(x, y2, marker='o', color='b', label='Speedup')
    plt.xlabel('Number of processes')
    plt.ylabel('Speedup')
    plt.title(f'Dependence of speedup on the number of processes for matrices ({dimension}, {dimension})')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'speedup_{dimension}.png')
    plt.show()


def plot_efficency(df, dimension):
    plt.figure(figsize=(15, 10))

    x = df['Processes']
    y1 = df['Amdahl efficency']
    y2 = df['Efficency']
    plt.plot(x, y1, marker='o', color='g', label='Amdahl efficency')
    plt.plot(x, y2, marker='o', color='b', label='Efficency')
    plt.xlabel('Number of processes')
    plt.ylabel('Efficency')
    plt.title(f'Dependence of efficency on the number of processes for matrices ({dimension}, {dimension})')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'efficency_{dimension}.png')
    plt.show()


def plot_scalability(*dfs):
    plt.figure(figsize=(15, 10))

    x = dfs[0][0]['Processes']
    for df in dfs:
        dimension = df[1]
        y = df[0]['Speedup']
        plt.plot(x, y, marker='o', label=f'Speedup for matrices ({dimension}, {dimension})')
    plt.xlabel('Number of processes')
    plt.ylabel('Speedup')
    plt.title('Scalability')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('scalability.png')
    plt.show()


def run_experiment(dimension):
    for pprocess in range(0, 8):
        run_matrix_computation(2 ** pprocess, dimension)

    times_df = read_csv(f'times_{dimension}.csv')
    plot_times(times_df, dimension)

    statistics_df = compute_statistics(times_df)
    print(f'{statistics_df}\n')
    statistics_df.to_csv(f'statistics_{dimension}.csv', index=False)
    plot_speedup(statistics_df, dimension)
    plot_efficency(statistics_df, dimension)

    return statistics_df


def main():
    run_experiment(500)

    dimension1000 = 1000
    statistics_df1000 = run_experiment(dimension1000)

    dimension2000 = 2000
    statistics_df2000 = run_experiment(dimension2000)

    plot_scalability((statistics_df1000, dimension1000),
                     (statistics_df2000, dimension2000))


if __name__ == '__main__':
    main()
