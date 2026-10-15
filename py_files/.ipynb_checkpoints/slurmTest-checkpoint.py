import matplotlib.pyplot as plt

def main():
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    
    plt.scatter(x, y)
    plt.savefig("slurmGraph.png")

if __name__ == '__main__':
    main()
