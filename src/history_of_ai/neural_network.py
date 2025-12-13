"""
By:             Lu Hou Yang
Last updated:   13th Dec 2025

A minimal neural network, to learn the fundamentals.
"""

import numpy as np


def main():
    # inputs = [Thirsty, Hungry]
    inputs = [0.7, 0.1]

    # weights = [
    #   [Thirsty-Refregirator, Hungry-Refregerator],
    #   [Thirsty-Cookie, Hungry-Cookie]
    # ]
    weights = [[0.9, 0.1], [0.1, 0.9]]

    output_threshold = [0.5, 0.5]
    output_decision = np.array(["Go To Refregirator", "Go To Cookie Jar"])

    # Calculations
    output = np.dot(inputs, weights)

    # Softmax
    # output = np.e**output / np.sum(np.e**output)

    print(output)
    print(output_decision[output >= output_threshold])


if __name__ == "__main__":
    main()
