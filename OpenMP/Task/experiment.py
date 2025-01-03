import subprocess
import pandas as pd
import matplotlib.pyplot as plt


def run_matrix_computation(dimension):
    sp = subprocess.run(
        ['g++', '-fopenmp', '-o', 'matrix_computation', 'matrix_computation.cpp'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if sp.returncode != 0:
        raise ChildProcessError('Matrix computation compilation error')

    sp = subprocess.run(
        ['./matrix_computation'] + [str(dimension)],
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

    x = df['Threads']
    y = df['Average']
    plt.plot(x, y, marker='o', color='b', label='Average')
    plt.xlabel('Number of threads')
    plt.ylabel('Time (sec.)')
    plt.title(f'Dependence of time on the number of threads for matrices ({dimension}, {dimension})')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'times_{dimension}.png')
    plt.show()


def plot_speedup(df, dimension):
    plt.figure(figsize=(15, 10))

    x = df['Threads']
    y1 = df['Amdahl speedup']
    y2 = df['Speedup']
    plt.plot(x, y1, marker='o', color='g', label='Amdahl speedup')
    plt.plot(x, y2, marker='o', color='b', label='Speedup')
    plt.xlabel('Number of threads')
    plt.ylabel('Speedup')
    plt.title(f'Dependence of speedup on the number of threads for matrices ({dimension}, {dimension})')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'speedup_{dimension}.png')
    plt.show()


def plot_efficency(df, dimension):
    plt.figure(figsize=(15, 10))

    x = df['Threads']
    y1 = df['Amdahl efficency']
    y2 = df['Efficency']
    plt.plot(x, y1, marker='o', color='g', label='Amdahl efficency')
    plt.plot(x, y2, marker='o', color='b', label='Efficency')
    plt.xlabel('Number of threads')
    plt.ylabel('Efficency')
    plt.title(f'Dependence of efficency on the number of threads for matrices ({dimension}, {dimension})')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'efficency_{dimension}.png')
    plt.show()


def plot_scalability(*dfs):
    plt.figure(figsize=(15, 10))

    x = dfs[0][0]['Threads']
    for df in dfs:
        dimension = df[1]
        y = df[0]['Speedup']
        plt.plot(x, y, marker='o', label=f'Speedup for matrices ({dimension}, {dimension})')
    plt.xlabel('Number of threads')
    plt.ylabel('Speedup')
    plt.title('Scalability')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('scalability.png')
    plt.show()


def run_experiment(dimension):
    run_matrix_computation(dimension)

    times_df = read_csv(f'times_{dimension}.csv')
    plot_times(times_df, dimension)

    statistics_df = read_csv(f'statistics_{dimension}.csv')
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
