---
layout: archive
title: ""   
permalink: /SFCO-CN/
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

## 阶梯函数约束优化
---

<p style="line-height: 1;"></p>
\begin{equation}
\min_{\mathbf{x}\in\mathbb{R}^{K}} ~~  f(\mathbf{x}),~~~~ \mbox{s.t.}~~ \parallel\mathbf{G}(\mathbf{x})\parallel_0^+\leq s,~\mathbf{x}\in \Omega  \tag{SFCO}
\end{equation}

<div style="text-align:justify;">
其中， 矩阵 $\mathbf{G}(\mathbf{x})\in\mathbb{R}^{M \times N}$ 的第 $(i,j)$ 个元素为 $G_{ij}(\mathbf{x})$，函数 $f:\mathbb{R}^{K}\rightarrow \mathbb{R}$ 和 $G_{ij}:\mathbb{R}^{K}\rightarrow \mathbb{R}$ 连续可微，最好二次连续可微，正整数 $s\ll n$，集合 $\Omega\subseteq\mathbb{R}^{K}$ 闭凸。度量 $\|\mathbf{Z}\|_0^+$ 计算矩阵 $\mathbf{Z}\in\mathbb{R}^{M \times N}$ 中含有正元素的列的个数，即
  \begin{equation}\|\mathbf{Z}\|_0^+= \mathrm{step}\Big(\max_{i=1,\ldots,M} Z_{i1}\Big)+\cdots+\mathrm{step}\Big(\max_{i=1,\ldots,M} Z_{iN}\Big)\nonumber\end{equation}
这里，$\mathrm{step}$ 是阶梯（又称 0/1 损失）函数，定义为：$\mathrm{step}(t)=1$ 当 $t>0$；否则 $\mathrm{step}(t)=0$。当 $M=1$，矩阵 $\mathbf{Z}$ 退化成向量 $\mathbf{z}\in\mathbb{R}^{N}$，如果令 $\mathbf{z}_+=$ $(\max\{0,z_1\}$ $\ldots$ $\max\{0,z_N\})^\top$ 以及零范数 $\parallel\mathbf{z}\parallel_0$ 计算 $\mathbf{z}$ 中非零元个数，则有  
  \begin{equation*}\|\mathbf{z}\|_0^+= \mathrm{step}(z_1)+\cdots+\mathrm{step}(z_N)=\|\mathbf{z}_+\|_0\end{equation*}
目前， 集合 $\Omega$ 可以是以下集合之一：    
</div> 

- 球: $\lbrace\mathbf{x}: \parallel\mathbf{x}\parallel^2\leq r\rbrace$，其中 $r>0$: 
- 半空间: $\lbrace\mathbf{x}: \mathbf{a}^T\mathbf{x}\leq b\rbrace$，其中 $\mathbf{a}\in\mathbb{R}^{K}$ 和 $b\in\mathbb{R}$
- 超平面: $\lbrace\mathbf{x}: \mathbf{A} \mathbf{x}=  \mathbf{b}\rbrace$，其中 $\mathbf{A}\in\mathbb{R}^{S\times K}$ 和 $ \mathbf{b}\in\mathbb{R}^{S}$
- 盒子:  $\lbrace\mathbf{x}: l\leq x_i \leq u, i=1,\ldots,K\rbrace$，其中 $l \leq u$ 分别可以取值 $-\infty$ 和 $+\infty$。因此，当 $l=-\infty$ 和 $u=+\infty$， $\Omega$ 为无约束；当 $l=0$ 和 $u=+\infty$， $\Omega$ 为非负约束
 
<!-- ## <span style="color:#8C8C8C"> The solver and its demonstration </span> -->

---
<div style="text-align:justify;"> 
程序包 - <a style="font-size: 16px; font-weight: bold;color:#006DB0" href=" " target="_blank">SFCOpack-Matlab</a> 或 <a style="font-size: 16px; font-weight: bold;color:#006DB0" href=" " target="_blank">SFCOpack-Python</a>（点击下载）提供了 1 个求解器，其核心算法来自以下文章：
</div>

> <b style="font-size:14px;color:#777777">SNSCO</b> - <span style="font-size: 14px"> S Zhou, L Pan, N Xiu,  and G  Li, A 0/1 constrained optimization solving sample average approximation for chance constrained programming, MOR, 2024. </span>

<!--
- <a style="font-size: 14px;color:#000000" href="https://jmlr.org/papers/v22/19-026.html" target="_blank"> S Zhou, N Xiu and H  Qi, Global and quadratic convergence of Newton hard-thresholding pursuit, *J Mach Learn Res*, 22:1−45, 2021.</a>
- <a style="font-size: 14px;color:#000000" href="https://www.sciencedirect.com/science/article/pii/S1063520322000458" target="_blank"> S Zhou, Gradient projection newton pursuit for sparsity constrained optimization, *Appl Comput Harmon Anal*, 61:75-100, 2022.</a> 
- <a style="font-size: 14px;color:#000000" href="http://www.yokohamapublishers.jp/online2/oppjo/vol13/p325.html" target="_blank"> L Pan, S Zhou, N Xiu, and H Qi, A convergent iterative hard thresholding for nonnegative sparsity optimization, *Pac J Optim*, 13:325-353, 2017.</a>  

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
 -->
