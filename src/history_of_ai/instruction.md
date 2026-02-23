# Simplest Neural Network

For this section we are using only [numpy](https://numpy.org/) library, an industrial and academic standard library for data manipulation (arrays, matrix operations, etc.)

## Basic Python program

This is the most common begining project structure

```python
# library imports
import numpy as np

# main logic / loop
def main():
    pass

# program start point
if __name__ == "__main__":
    main()
```

## Basic Components in a Neural Network

1. Inputs

    ```python
    # inputs = [Thirsty, Hungry]
    inputs = [0.7, 0.1]
    ```

1. Weights

    ```python
    # weights = [
    #   [Thirsty-Refrigerator, Hungry-Refrigerator],
    #   [Thirsty-Cookie, Hungry-Cookie]
    # ]
    weights = [[0.9, 0.1], [0.1, 0.9]]
    ```

1. Outputs

    ```python
    output_decision = np.array(["Go To Refrigerator", "Go To Cookie Jar"])
    ```

1. Decision Function

    ```python
    output_threshold = [0.5, 0.5]
    ```

1. Feed-Forward

    ```python
    output = np.dot(inputs, weights)
    ```

1. Print Output

    ```python
    print(output)
    print(output_decision[output >= output_threshold])
    ```

Put everything together in the main function and run it >.<
