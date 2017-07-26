<script type="text/javascript"
	src="//cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML,
	/javascripts/MathJaxLocal.js
">
</script>

# Nonnegative Matrix Factorization

Nonnegative matrix factorization is a computational technique of dimensional reduction of a given data to uncover the latent factors embedded in higher dimensions. Unlike traditional matrix decomposition methods such as SVD and full rank decomposition, the non-negativity constraint imposed by NMF is useful for learning part-based representations. Secondly, since that in many real world applications such as image and face recognition, the data matrices people are dealing with are usually nonnegative, and that intuitively parts are generally combined additively (not subtracted as what many face recognition problems using SVD do, which generate not nonnegative eigenfaces) to form a whole picture and physiological principles assume that humans learn objects as part-based, the non-negativity thereby enhances meaningful interpretations of information given by the data matrix and is applicable to real world problems.  

## Definition

Suppose a nonnegative matrix $A \in R^{M \times N}$ is given. NMF returns the decomposed representation of $A$ with nonnegative matrix $W$ and $H$ by solving a nonconvex optimization problem defined with Frobenius norm.

$$min \; f(W, \; H) = {\left \lVert A-WH^T \right \rVert}^2_F $$ $$subject \ to \ W \ge 0, \ H \ge 0$$

Since this optimization problem is nonconvex, only the local minimum should be expected from any good algorithm.

## Algorithms

* **HALS**

![](/pic/1.PNG)
![](/pic/2.PNG)
