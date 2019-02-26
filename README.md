## BME 495 HW 3
### Parth Doshi - doshi8@purdue.edu

## AND gate
### Manual Weights
`
[
    [-15, 10, 10]
]
`

### Learnt weights
`
[
    [-6.1323, 4.0198, 4.0198]
]
`

## OR gate
### Manual Weights
`
[
    [-5, 10, 10]
]
`

### Learnt Weights
`
[
    [-1.7215, 4.0207, 3.9974]
]
`

## NOT gate
### Manual weights
`
[
    [10, -20]
]
`

### Learnt Weights
`
[
    [2.0885, -4.4193]
]
`

## XOR gate
### Manual weights
`
[
    [
        [-5, 9],
        [10, -6],
        [10, -6]
    ],
    [
        [-15],
        [10],
        [10]
    ]
]
`

### Learnt Weights
`
[
    [
        [-2.5798, 5.6489],
        [6.0893, -3.7905],
        [6.1455, -3.7991]
    ],
    [
        [-8.6849],
        [5.8674],
        [6.0890]
    ]
]
`

## Conclusions
We can see that our learnt weights are pretty similar to our manually tuned weights if we take ratios between weights into account.

One caveat that I found was the XOR doesn't always converge unless the random initialization is favorable. Changing the random initialization from a normal distribution to a xavier normal distribution increased the percentage of convergence. To make sure the `train` method would converge, I put a while loop in xor's train function that rerandomizes the weights every run. Also, I found XOR converged to many different parameter values that worked while my other functions almost always picked the same weights every time.