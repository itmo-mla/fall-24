import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from core.optim import MomentumSGD
from core.loss import Lin, RMSE
from core.regular import L2
from read import read_data
from core.weights import RandomWeightGenerator
from core.utils import accuracy
from core.model import LinearClassificator
    
if __name__ == "__main__":
    X, y = read_data('/Users/ilyadanilenko/Documents/GitHub/fall-24/students/ie-danilenko/lab4/source/dataset/Iris.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    optim = MomentumSGD(1e-5, 0.5)
    regu = L2(0.3)
    weight_gen = RandomWeightGenerator()

    model = LinearClassificator([Lin, RMSE], optim, regu, weight_gen)
    model.train(X_train, y_train, X_test, y_test, epochs=50)
    plt.show()

    y_pred = model.predict(X_test)
    print(y_pred)
    print("STAT:")
    print(accuracy(y_pred, y_test))