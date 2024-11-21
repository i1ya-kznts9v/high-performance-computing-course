import subprocess
import pandas as pd
import matplotlib.pyplot as plt


def run_matrix_computation(dimension):
    result = subprocess.run(
        ['g++', '-fopenmp', '-o', 'matrix_computation', 'matrix_computation.cpp'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode != 0:
        raise ChildProcessError("Matrix computation compilation error")

    result = subprocess.run(
        ['./matrix_computation'] + [str(dimension)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode != 0:
        raise ChildProcessError("Matrix computation error")


def read_csv(file_name):
    df = pd.read_csv(file_name)
    print(f'{df}\n')
    return df


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
        y = df[0]['Speedup']
        dimension = df[1]
        plt.plot(x, y, marker='o', label=f'Speedup for matrices ({dimension}, {dimension})')
    plt.xlabel('Number of threads')
    plt.ylabel('Speedup')
    plt.title('Scalability')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('scalability.png')
    plt.show()


def main():
    dimension100 = 100
    run_matrix_computation(dimension100)
    df100 = read_csv(f'results_{dimension100}.csv')
    plot_speedup(df100, dimension100)
    plot_efficency(df100, dimension100)

    dimension1000 = 1000
    run_matrix_computation(dimension1000)
    df1000 = read_csv(f'results_{dimension1000}.csv')
    plot_speedup(df1000, dimension1000)
    plot_efficency(df1000, dimension1000)

    dimension2000 = 2000
    run_matrix_computation(dimension2000)
    df2000 = read_csv(f'results_{dimension2000}.csv')
    plot_speedup(df2000, dimension2000)
    plot_efficency(df2000, dimension2000)

    plot_scalability((df1000, dimension1000), (df2000, dimension2000))


if __name__ == "__main__":
    main()
