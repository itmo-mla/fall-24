from core.model import LinearClassificator
from core.weights import StaticWeightGenerator
from read import read_data
from core.loss import Lin
import matplotlib.pyplot as plt

if __name__ == "__main__":
    X, y = read_data('/Users/ilyadanilenko/Documents/GitHub/fall-24/students/ie-danilenko/lab4/source/dataset/Iris.csv')
    dataset = list(zip(X, y))

    model = LinearClassificator([], None, None, StaticWeightGenerator(1))

    margins = Lin.get(X, y, model.layers[0].weights).flatten()
    
    plt.bar(range(len(margins)), sorted(margins), color=['red' if m < 0 else 'green' for m in sorted(margins)])
    plt.show()