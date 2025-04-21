---
layout: archive
title: ""   
permalink: /SFCO/
author_profile: true
---

<style>
a:link {
  text-decoration: none;
}

a:visited {
  text-decoration: none;
}

a:hover {
  text-decoration: underline;
}

a:active {
  text-decoration: underline;
}
</style>

 

##  <span style="color:#8C8C8C"> Step function-constrained optimization</span> 
---

<p style="line-height: 1;"></p>
\begin{equation}
\min_{\mathbf{x}\in\mathbb{R}^{K}} ~~  f(\mathbf{x}),~~~~ \mbox{s.t.}~~ \parallel\mathbf{G}(\mathbf{x})\parallel_0^+\leq s,~\mathbf{x}\in \Omega  \tag{SFCO}
\end{equation}

<div style="text-align:justify;">
where $\mathbf{G}(\mathbf{x})=[G_{ij}(\mathbf{x})]_{1\leq i\leq M, 1\leq j \leq N}$, both $f:\mathbb{R}^{K}\rightarrow \mathbb{R}$ and $G_{ij}:\mathbb{R}^{K}\rightarrow \mathbb{R}$ are (preferably twice) continuously differentiable functions, $\Omega\subseteq\mathbb{R}^{K}$ is a closed and convex set, $s\ll N$ is an integer, and $\|\mathbf{Z}\|_0^+$ counts the number of columns in  $\mathbf{Z}\in\mathbb{R}^{M \times N}$ having positive entries, i.e., 
  \begin{equation}\|\mathbf{Z}\|_0^+=\sum_{j=1}^N \ell_{0/1}\left(\max_{i=1,2,\ldots,M} Z_{ij}\right)\nonumber\end{equation}
  Here, $\ell_{0/1}(t)$ is the step function (or 0/1 loss function) defined by $\ell_{0/1}(t)=1$ if $t>0$ and $\ell_{0/1}(t)=0$ otherwise. For the case of $M=1$, let $(\mathbf{z})_+=(\max\{0,z_1\},\ldots,\max\{0,z_N\})^\top$ and  $\parallel\mathbf{z}\parallel_0$ denote the $\ell_0$ norm that counts the number of nonzero entries in $\mathbf{z}$. One can observe that
  \begin{equation}\parallel\mathbf{z}\parallel_0^+=\sum_{j=1}^N \ell_{0/1}\left(z_{j}\right)=\parallel(\mathbf{z})_+\parallel_0\nonumber\end{equation}
</div>
 
<!-- ## <span style="color:#8C8C8C"> The solver and its demonstration </span> -->

---
<div style="text-align:justify;"> 
The solver can be download here - <a style="font-size: 16px; font-weight: bold;color:#006DB0" href="https://github.com/ShenglongZhou/SNSCO" target="_blank">SNSCO</a>,
which was developed from the following paper:
</div>

> <span style="font-size: 14px"> S Zhou, L Pan, N Xiu,  and G  Li, A 0/1 constrained optimization solving sample average approximation for chance constrained programming, Math Oper Res, 2024. </span>

<!--
- <a style="font-size: 14px;color:#000000" href="https://jmlr.org/papers/v22/19-026.html" target="_blank"> S Zhou, N Xiu and H  Qi, Global and quadratic convergence of Newton hard-thresholding pursuit, *J Mach Learn Res*, 22:1−45, 2021.</a>
- <a style="font-size: 14px;color:#000000" href="https://www.sciencedirect.com/science/article/pii/S1063520322000458" target="_blank"> S Zhou, Gradient projection newton pursuit for sparsity constrained optimization, *Appl Comput Harmon Anal*, 61:75-100, 2022.</a> 
- <a style="font-size: 14px;color:#000000" href="http://www.yokohamapublishers.jp/online2/oppjo/vol13/p325.html" target="_blank"> L Pan, S Zhou, N Xiu, and H Qi, A convergent iterative hard thresholding for nonnegative sparsity optimization, *Pac J Optim*, 13:325-353, 2017.</a> -->

---
<div style="text-align:justify;">  
Note that <b style="font-size:14px;color:#777777">NHTP</b> and <b style="font-size:14px;color:#777777">GPNP</b> are second-order methods, which require the gradient and Hessian of $f$. <b style="font-size:14px;color:#777777">IIHT</b> is a first-order method that only requires the gradient. Below is a demonstration of how to define the gradient and Hessian for <b style="font-size:14px;color:#777777">NHTP</b>.
</div>

<p style="line-height: 1;"></p>

```ruby
function [out1,out2] = funCS(x,T1,T2,data)

    if  isempty(T1) && isempty(T2) 
        Tx   = find(x); 
        Axb  = data.A(:,Tx)*x(Tx)-data.b;
        out1 = norm(Axb,'fro')^2/2;               %objective 
        if  nargout == 2
            out2    = (Axb'*data.A)';             %gradient
        end
    else        
        AT = data.A(:,T1); 
        if  length(T1)<2000
            out1 = AT'*AT;                        %subHessian containing T1 rows and T1 columns
        else
            out1 = @(v)( (AT*v)'*AT )';      
        end       
        if  nargout == 2
            out2 = @(v)( (data.A(:,T2)*v)'*AT )'; %subHessian containing T1 rows and T2 columns
        end       
    end     
end
```
