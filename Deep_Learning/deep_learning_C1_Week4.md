# Week 4

## Deep L-layer Neural Network

### Deep Neural Network Notation



<figure>
    <img src="../images/deep_NN.png" width="550"/>
    <figcaption>4 layer NN</figcaption>
</figure>

$L=4$ (# layers)

$n^{[l]}=$ #units in layer $l$

$n^{[0]}=n_{x}=3,n^{[1]}=5,n^{[2]}=5,n^{[3]}=3,n^{[4]}=1$

$a^{[l]} =$ activation in layer $l$

$a^{[l]}=g'^{[l]}(z^{[l]})$

$\boxed{a^{[0]}=x\\\hat{y}=a^{[L]}}$

$W^{[l]}=$ weights for $z^{[l]}$

$b^{[l]}=$ biases for $z^{[l]}$

### Forward Propagation for layer  $l$

$$
\begin{split}
&\text{Input }a^{[l-1]}\\
&\text{Output }a^{[l]},\text{cache }(z^{[l]},W^{[l]},b^{[l]})\\
&z^{[l]}=W^{[l]}a^{[l-1]}+b^{[l]}\\
&a^{[l]}=g^{[l]}(z^{[l]})
\end{split}
$$

---

$$
\begin{split}
&\text{Vectorized:}\\
&Z^{[l]}=W^{[l]}A^{[l-1]}+b^{[l]}\\
&A^{[l]}=g^{[l]}(Z^{[l]})
\end{split}
$$

---

$$
X=A^{[0]}\rarr
\boxed
{
ReLU
}\rarr
\boxed
{
ReLU
}\rarr
\boxed
{
ReLU
}\rarr
\boxed
{
Sigmoid
}\rarr
\hat{Y}
$$

### Backward Propagation for layer  $l$

$$
\begin{split}
&\text{Input }da^{[l]}\\
&\text{Output }da^{[l-1]},dW^{[l]},db^{[l]}\\
\text{}\\
&dz^{[l]}=da^{[l]}*g'^{[l]}(z^{[l]})\\
&dW^{[l]}=dz^{[l]}a^{[l-1]}\\
&db^{[l]}=dz^{[l]}\\
&da^{[l-1]}=W^{[l]T}dz^{[l]}\\
\end{split}
$$

---

$$
\begin{split}
Vectorized:\\
dZ^{[l]}&=dA^{[l]}*g'^{[l]}(Z^{[l]})\\
dW^{[l]}&=\frac{1}{m}dZ^{[l]}A^{[l-1]T}\\
db^{[l]}&=\frac{1}{m}np.sum(dZ^{[l]},axis=1)\\
dA^{[l-1]}&=W^{[l]T}dZ^{[l]}
\end{split}
$$

---

$$
\boxed
{
dA^{[2]}=W^{[3]T}dZ^{[3]}\\
dZ^{[2]}=dA^{[2]}*g'^{[2]}(Z^{[2]})
}
\larr 
\boxed
{
dA^{[3]}=W^{[L]T}dZ^{[L]}\\
dZ^{[3]}=dA^{[3]}*g'^{[3]}(Z^{[3]})
}
\larr 
\boxed
{
dA^{[L]}=-\frac{Y}{A^{[L]}}+\frac{1-Y}{1-A^{[L]}}\\
dZ^{[L]}=dA^{[L]}*g'^{[L]}(Z^{[l]})
}
\larr 
L(A^{[L]},Y)
$$



### Forward Propagation in a Deep Neural Network

