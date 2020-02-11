Week 3

## Review Logistic Regression

1. Forward Propagation

![logistic regression](C:\Users\xiaohai\AppData\Roaming\Typora\typora-user-images\image-20200209222805675.png)



2. Details of a single node

   <img src="C:\Users\xiaohai\AppData\Roaming\Typora\typora-user-images\image-20200209223358151.png" width="400"/>



## Neural Network

1. Overview

<img src="C:\Users\xiaohai\AppData\Roaming\Typora\typora-user-images\image-20200209223141492.png" width=500/>



2. Define ```W_l, b_l, Z_l, A_l```

<img src="C:\Users\xiaohai\AppData\Roaming\Typora\typora-user-images\image-20200209224016523.png" width="550"/>

__Attention!__ ```z, a``` are lower case since they are calculated by only one example which is ```x```.
$$
a^{[0]}=x
$$

$$
a^{[0]}=\begin{bmatrix}
x_{1}\\
x_{2}\\
x_{3}\\
\vdots
\end{bmatrix}
$$

$$
W^{[l]}\text{'s row number is dependent on the number of hidden units at that layer}
$$

$$
W^{[l]}\text{'s column number is dependent on the row number of the previous layer}
$$




$$
W^{[l]}=\begin{bmatrix}
w_{1}^{[l]T}\\
w_{2}^{[l]T}\\
w_{3}^{[l]T}\\
w_{4}^{[l]T}\\
w_{5}^{[l]T}\\
\end{bmatrix}
$$


```python
# input x is also a_0
# a_0.shape = (n_x,1)
a_0 == x

# W_l stack w.T horizontally (w.T(1) subsript refer to the image above)
# W_l.Shape = (n_l,n_(l-1)) W_l.Column is dependent on the previous layer's unit number

# b_l stack b horiozontally
# b_l.Shape = (n_l,1) b_l.Row is dependent on the number of unit at the current layer

# z_l stack z horizontally (z)
# z_l.Shape = (n_l,1)

# a_l stack a horizontally (a(1) subsript check the image above)
# a_l.Shape = (n_l,1)

```

3. Compute y_hat or a_L

   <img src="C:\Users\xiaohai\AppData\Roaming\Typora\typora-user-images\image-20200209231918479.png" width=550/>

```python
a_0 = x
z_1 = W_1 * a_0 + b_1
a_1 = sigmoid(z_1)

z_2 = W_2 * a_1 +b_2
a_2 = sigmoid(z_2)
```





4. Vectorizing across multiple examples and define ```Z_l, A_l```

   

   **Note:** ```(i)``` refers to example index, ```[l]```refers to layer index

   <img src="C:\Users\xiaohai\AppData\Roaming\Typora\typora-user-images\image-20200209233431111.png" width=550/>

```python
# X.Shape = (n_x,m)
# W_l.Shape = (n_l,n_(l-1))
# b_l.Shape = (n_l,1)
# Z_l.Shape = (n_l,m)
# A_l.Shape = (n_l,m)
```



## Activation Functions

1. Sigmoid Function
   $$
   sigmoid(z) = \frac{1}{1+e^{-z}}
   $$

2. Tanh Function (Hyperbolic tangent)
   $$
   tanh(z)=\frac{e^{z}-e^{-z}}{e^{z}+e^{-z}}
   $$

3. RELU Function (Rectified Linear Unit)
   $$
   relu(z)=max(0,z)
   $$

## Derivatives of activation functions

1. Sigmoid Activation Function
   $$
   g(z)=\sigma(z)=\frac{1}{1+e^{-z}}\\
   \begin{split}
   g'(z) & =\frac{1}{1+e^{-z}}(1-\frac{1}{1+e^{-z}})\\
   & =g(z)(1-g(z))
   \end{split}
   $$
   

2. Tanh Activation Function
   $$
   g(z)=tanh(z)=\frac{e^{z}-e^{-z}}{e^{z}+e^{-z}}\\
   \begin{split}
   g'(z)=\frac{dg(z)}{dz}&=\text{slop of g(z) at z}\\
   & = 1 - (tanh(z))^2\\
   & = 1 - g(z)^2
   \end{split}
   $$

3. ReLU and Leakly ReLU
   $$
   ReLU\\
   \begin{split}
   g(z)&=Max(0,z)\\
   g'(z)&=\begin{cases}
   0 & \text{if }z<0\\
   1 & \text{if }z\geq0
   \end{cases}
   \end{split}
   $$
   

$$
Leakly ReLU\\
\begin{split}
g(z)&=Max(0.01z,z)\\
g'(z)&=\begin{cases}
0.01 & \text{if }z<0\\
1 & \text{if }z\geq0
\end{cases}
\end{split}
$$

