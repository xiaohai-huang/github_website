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

### Activation derivatives table

| Function Name      | Definition                                  | Derivative                                                   |
| ------------------ | ------------------------------------------- | ------------------------------------------------------------ |
| Sigmoid            | $\sigma(z)=\frac{1}{1+e^{-z}}$              | $\frac{d\sigma(z)}{dz}=\sigma(z)(1-\sigma(z))$               |
| Hyperbolic tangent | $tanh(z)=\frac{e^{z}-e^{-z}}{e^{z}+e^{-z}}$ | $\frac{dtanh(z)}{dz}=1 - (tanh(z))^2$                        |
| ReLU               | $ReLU(z)=Max(0,z)$                          | $\begin{split}\\ReLU'(z)&=\begin{cases}0 & \text{if }z<0\\1 & \text{if }z\geq0\end{cases}\end{split}$ |



## Gradient Descent For Neural Networks

### Forward Propagation

Given an input $X$ ``Shape=(n_x,m)``after a forward propagation we can obtain $\hat{Y}$``Shape=(1,m)``.
$$
\begin{split}
A^{[0]}&=X\\
Z^{[1]}&=W^{[1]}A^{[0]}+b^{[1]}\\
A^{[1]}&=g^{[1]}(Z^{[1]})\\
Z^{[2]}&=W^{[2]}A^{[1]}+b^{[2]}\\
A^{[2]}&=g^{[2]}(Z^{[2]})=\sigma(Z^{[2]})\\
\hat{Y}&=A^{[2]}
\end{split}
$$


### Backward Propagation

``Loss (Error) function`` is for a single example. It measures how well our ``parameters`` are doing for the example. ``Cost function`` is the **average loss** over the entire training set. The optimization strategies aim at minimizing the ``cost function``.

*Loss (error)*:
$$
L(\hat{y},y)=-(ylog(\hat{y})+(1-y)log(1-\hat{y}))
$$

*Cost function*: 
$$
J(W^{[1]},b^{[1]},W^{[2]},b^{[2]})=\frac{1}{m}\sum_{i=1}^{m}L(\hat{y},y)
$$

*Backward Propagation*
$$
\begin{split}
dA^{[2]}&=-\frac{Y}{A^{[2]}}+\frac{1-Y}{1-A^{[2]}}\\
dZ^{[2]}&=A^{[2]}-Y\\
dW^{[2]}&=\frac{1}{m}dZ^{[2]}A^{[1]T}\\
db^{[2]}&=\frac{1}{m}np.sum(dZ^{[2]},axis=1)\\
dA^{[1]}&=W^{[2]T}dZ^{[2]}\\
dZ^{[1]}&=dA^{[1]}*g^{[1]'}(Z^{[1]})\\
dW^{[1]}&=\frac{1}{m}dZ^{[1]}A^{[0]T}\\
		&=\frac{1}{m}dZ^{[1]}X^{T}\\
db^{[1]}&=\frac{1}{m}np.sum(dZ^{[1]},axis=1)
\end{split}
$$

---

### Backward Propagation for 1 hidden NN little summary

<img width="300" src="C:\Users\xiaohai\AppData\Roaming\Typora\typora-user-images\image-20200212105813146.png"/>

*Loss (error)*:
$$
L(\hat{y},y)=-(ylog(\hat{y})+(1-y)log(1-\hat{y}))
$$

$$
\begin{bmatrix}
x_{1}\\
w_{1}\\
x_{2}\\
w_{2}\\
x_{3}\\
w_{3}\\
b\\
\vdots
\end{bmatrix}
\rarr 
\boxed{z^{[1]}=W^{[1]}a^{[0]}+b^{[1]}}
\rarr
\boxed{a^{[1]}=g^{[1]}(z^{[1]})}
\rarr
\boxed{z^{[2]}=W^{[2]}a^{[1]}+b^{[2]}}
\rarr
\boxed{a^{[2]}=\sigma^{[2]}(z^{[2]})}
\\\rarr
\boxed{L(a^{[2]},y)}
$$



---

Compute partial derivatives $\frac{\partial L(a^{[2]},y)}{\partial z^{[1]}},\frac{\partial L(a^{[2]},y)}{\partial z^{[2]}}$
$$
\boxed 
{
\begin{split}
\frac{\partial L(a^{[2]},y)}{\partial z^{[1]}}&=\frac{\partial L(a^{[2]},y)}{\partial a^{[1]}}\frac{\partial a^{[1]}}{\partial z^{[1]}}\\
&=\frac{\partial L(a^{[2]},y)}{\partial a^{[1]}}g'^{[1]}(z^{[1]})
\end{split}
}
\larr
\boxed
{
\begin{split}
\frac{\partial L(a^{[2]},y)}{\partial a^{[1]}}&=\frac{\partial L(a^{[2]},y)}{\partial z^{[2]}}\frac{\partial z^{[2]}}{\partial a^{[1]}}\\
&=\frac{\partial L(a^{[2]},y)}{\partial z^{[2]}}W^{[2]}
\end{split}
}
\larr
\\
\larr
\boxed
{
\begin{split}
\frac{\partial L(a^{[2]},y)}{\partial z^{[2]}}&=\frac{\partial L(a^{[2]},y)}{\partial a^{[2]}}\frac{\partial a^{[2]}}{\partial z^{[2]}}\\
&=(-\frac{y}{a^{[2]}}+\frac{1-y}{1-a^{[2]}})(a^{[2]}(1-a^{[2]}))\\
&=a^{[2]}-y
\end{split}
}
\larr
\boxed
{
\frac{\partial L(a^{[2]},y)}{\partial a^{[2]}}=-\frac{y}{a^{[2]}}+\frac{1-y}{1-a^{[2]}}
}
$$


Compute partial derivatives $\frac{\partial L(a^{[2]},y)}{\partial W^{[1]}},\frac{\partial L(a^{[2]},y)}{\partial b^{[1]}},\frac{\partial L(a^{[2]},y)}{\partial W^{[2]}},\frac{\partial L(a^{[2]},y)}{\partial b^{[2]}}$

$$
\\
\boxed
{
\begin{split}
\frac{\partial L(a^{[2]},y)}{\partial W^{[1]}}&=\frac{\partial L(a^{[2]},y)}{\partial z^{[1]}}\frac{\partial z^{[1]}}{\partial W^{[1]}}\\
&=\frac{\partial L(a^{[2]},y)}{\partial z^{[1]}}a^{[0]}
\end{split}
}
\boxed
{
\begin{split}
\frac{\partial L(a^{[2]},y)}{\partial b^{[1]}}&=\frac{\partial L(a^{[2]},y)}{\partial z^{[1]}}\frac{\partial z^{[1]}}{\partial b^{[1]}}\\
&=\frac{\partial L(a^{[2]},y)}{\partial z^{[1]}}
\end{split}
}

\larr

\boxed
{
\frac{\partial L(a^{[2]},y)}{\partial z^{[1]}}
}

\\
\boxed
{
\begin{split}
\frac{\partial L(a^{[2]},y)}{\partial W^{[2]}}&=\frac{\partial L(a^{[2]},y)}{\partial z^{[2]}}\frac{\partial z^{[2]}}{\partial W^{[2]}}\\
&=\frac{\partial L(a^{[2]},y)}{\partial z^{[2]}}a^{[1]}
\end{split}
}
\boxed
{
\begin{split}
\frac{\partial L(a^{[2]},y)}{\partial b^{[2]}}&=\frac{\partial L(a^{[2]},y)}{\partial z^{[2]}}\frac{\partial z^{[2]}}{\partial b^{[2]}}\\
&=\frac{\partial L(a^{[2]},y)}{\partial z^{[2]}}
\end{split}
}

\larr

\boxed
{
\frac{\partial L(a^{[2]},y)}{\partial z^{[2]}}
}
$$
