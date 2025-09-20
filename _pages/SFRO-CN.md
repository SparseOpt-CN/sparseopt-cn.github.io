---
layout: archive
title: ""   
permalink: /SFRO-CN/
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


##  阶梯函数正则优化
---

<p style="line-height: 1;"></p>
\begin{equation}
\min_{\mathbf{x}\in\mathbb{R}^{n}} ~~  f(\mathbf{x}) + \lambda \parallel(\mathbf{B}\mathbf{x}+\mathbf{b})_+\parallel_0  \tag{SFRO}
\end{equation}

<div style="text-align:justify;">
其中，函数 $f:\mathbb{R}^{n}\rightarrow \mathbb{R}$ 二次连续可微，矩阵 $\mathbf{B}\in\mathbb{R}^{m\times n}$，向量 $\mathbf{b}\in\mathbb{R}^{m}$，罚参数 $\lambda>0$，零范数 $\|\mathbf{z}\|_0$ 表示 $\mathbf{z}$ 中非零元的个数。 记 $\mathbf{z}_+=$ $(\max\{0,z_1\}$ $\ldots$ $\max\{0,z_m\})^\top$，则 $\|\mathbf{z}_+\|_0$ 计算 $\mathbf{z}$ 中正元素的个数。该函数与阶梯（又称 0/1 损失）函数相关，其定义为：$\mathrm{step}(t)=1$ 当 $t>0$，否则 $\mathrm{step}(t)=0$。因此，$\|\mathbf{z}_+\|_0=$ $\mathrm{step}(z_1)+$ $\cdots$ $+\mathrm{step}(z_m)$。  
</div>
 
<!-- ## <span style="color:#8C8C8C"> The solver and its demonstration </span> -->

---
<div style="text-align:justify;"> 
程序包 - <a style="font-size: 16px; font-weight: bold;color:#006DB0" href="\files\SFROpack-Matlab.zip" target="_blank">SFROpack-Matlab</a> 和 <a style="font-size: 16px; font-weight: bold;color:#006DB0" href="\files\SFROpack-Python.zip" target="_blank">SFROpack-Python</a>（点击下载）提供了 1 个求解器，其核心算法来自以下文章：  
</div>

> <b style="font-size:14px;color:#777777">NM01</b> - <span style="font-size: 14px"> S Zhou, L Pan, N Xiu,  and H Qi, Quadratic convergence of smoothing Newton's method for 0/1 loss optimization, SIOPT, 31:3184–3211, 2021. </span>

---
<div style="text-align:justify;">  
求解器 $\texttt{NM01}$ 核心算法属于二阶方法，需要用到目标函数值、梯度和海瑟矩阵。基于 Matlab 语言（基于 Python语言，可进行类似定义）下面用1-比特压缩感知（1BCS）作为示例，展示如何为该求解器定义这些内容。 1BCS 问题的目标函数可以参考 <a style="font-size: 16px; font-weight: bold; color:#006DB0" href="https://sparseopt-cn.github.io/1BCS/" target="_blank">1BCS</a> 页面中的模型（<a style="font-size: 16px;color:#006DB0" href="https://sparseopt-cn.github.io/1BCS/" target="_blank">SFRO</a>）。下面的 MATLAB 代码定义了模型中目标函数值、梯度和海瑟矩阵，其中，函数句柄 $\texttt{func1BCS}$ 中的输入 $\texttt{x}$ 和 $\texttt{key}$ 为两个变量，其他输入 $\texttt{eps}$、$\texttt{q}$、$\texttt{A}$ 和 $\texttt{c}$ 为模型（<a style="font-size: 16px;color:#006DB0" href="https://sparseopt-cn.github.io/1BCS/" target="_blank">SFRO</a>）中给定的参数和数据。这里，字符串变量 $\texttt{key}$ 用于指定计算内容：当 $\texttt{key}$='$\texttt{f}$' 时计算目标函数值；当 $\texttt{key}$='$\texttt{g}$' 时计算梯度；当 $\texttt{key}$='$\texttt{h}$' 时计算海瑟矩阵。当 $\texttt{key}$='$\texttt{a}$' 时，会额外计算一个用户自定义函数。在此示例中，计算的是 1BCS 问题的准确率。这使得用户能够在优化过程中监控自定义指标。  
</div>
<p style="line-height: 1;"></p>

```ruby
function out = func1BCS(x,key,eps,q,A,c) 
    switch key   
        case 'f';  out = sum((x.^2+eps).^(q/2));
        case 'g';  out = q*x.*(x.^2+eps).^(q/2-1); 
        case 'h';  x2  = x.*x; out = diag(((x2+eps).^(q/2-2)).*((q-1)*x2+eps)); 
        case 'a';  acc = @(var)nnz(sign(A*var)-c); out = 1-acc(x)/length(c);
        otherwise; out = []; % 'Otherwise' is REQIURED if no key='a'
    end    
end
```

<div style="text-align:justify;">  
如果不需要额外的函数，用户只需定义目标函数值、梯度和海瑟矩阵，然后省略情况 $\texttt{key}$='$\texttt{a}$'，如下所示。注意，当情况 $\texttt{key}$='$\texttt{a}$' 被省略时，用户必须保留情况 $\texttt{otherwise}$。
</div>
<p style="line-height: 1;"></p>

```ruby
function out = func1BCS(x,key,eps,q) 
    switch key   
        case 'f';  out = sum((x.^2+eps).^(q/2));
        case 'g';  out = q*x.*(x.^2+eps).^(q/2-1); 
        case 'h';  x2  = x.*x; out = diag(((x2+eps).^(q/2-2)).*((q-1)*x2+eps)); 
        otherwise; out = [];  
    end    
end
```

<div style="text-align:justify;">
对于 1BCS 模型（<a style="font-size: 16px;color:#006DB0" href="https://sparseopt-cn.github.io/1BCS/" target="_blank">SFRO</a>），定义好如上函数后，就可以调用 $\texttt{NM01}$ 来求解该问题。用户只需指定 ($\texttt{func}$, $\texttt{B}$, $\texttt{b}$, $\texttt{lam}$, $\texttt{pars}$)，然后运行求解器即可。
</div>

<p style="line-height: 1;"></p>

```ruby
% Solving 1 bit compressive sensing using randomly generated data 
clc; close all; clear all; addpath(genpath(pwd));

n            = 1000; 
m            = ceil(0.5*n);
s            = ceil(0.01*n);                      % sparsity level
r            = 0.01;                              % flipping ratio
nf           = 0.05;                              % noisy ratio
[A,c,co,xo]  = random1bcs('Ind',m,n,s,nf,r,0.5);  % data generation

func         = @(x,key)func1BCS(x,key,1e-5,0.5,A,c);
B            = (-c).*A;
b            = (n*8e-5)*ones(m,1);
lam          = 1;
pars.tau     = 1;  
pars.strict  =(n<=2000); 
out          = NM01(func, B, b, lam, pars); 
x            = refine(out.sol,s,A,c);

PlotRecovery(xo,x,[950,500,500,250],1)
fprintf(' Computational time:    %.3fsec\n',out.time);
fprintf(' Signal-to-noise ratio: %.2f\n',-20*log10(norm(x-xo)));
fprintf(' Hamming distance:      %.3f\n',nnz(sign(A*x)-c)/m)
fprintf(' Hamming error:         %.3f\n',nnz(sign(A*x)-co)/m)
```

<div style="text-align:justify;">
Matlab 版求解器 $\texttt{NM01}$ 的输入与输出（Python 版的输入与输出类似）说明如下，其中输入参数 ($\texttt{func}$, $\texttt{B}$, $\texttt{b}$, $\texttt{lam}$) 为必需项。$\texttt{pars}$ 中的参数为可选项，但设置某些参数可能会提升求解器的性能和解的质量。 
</div>

<p style="line-height: 1;"></p>

```ruby
function out = NM01(func,B,b,lam,pars)
% -------------------------------------------------------------------------
% This code aims at solving the support vector machine with form
%
%       min  f(x) + lam * ||(Bx+b)_+||_0
%
% where f is twice continuously differentiable
% lam > 0, B\in\R^{m x n}, b\in\R^{m x 1}
% (z)_+ = (max{0,z_1},...,max{0,z_m})^T
% ||(z)_+ ||_0 counts the number of positive entries of z
% -------------------------------------------------------------------------
% Inputs:
%   func: A function handle defines (objective,gradient,Hessain) (REQUIRED)
%   B   : A matrix \R^{m x n}                                    (REQUIRED)      
%   b   : A vector \R^{m x 1}                                    (REQUIRED)
%   lam : The penalty parameter                                  (REQUIRED)
%   pars: Parameters are all OPTIONAL
%         pars.x0     -- The initial point             (default zeros(n,1))
%         pars.tau    -- A useful paramter                   (default 1.00)
%         pars.mu0    -- A smoothing parameter               (default 0.01)
%         pars.maxit  -- Maximum number of iterations        (default 1000)  
%         pars.tol    -- Tolerance of halting conditions   (1e-7*sqrt(n*m)) 
%         pars.strict -- = 0, loosely meets halting conditions  (default 0)
%                        = 1, strictly meets halting conditions  
%                        pars.strict=1 is useful for low dimensions                           
% -------------------------------------------------------------------------
% Outputs:
%   out.sol:  The solution 
%   out.obj:  The objective function value
%   out.time: CPU time
%   out.iter: Number of iterations
% -------------------------------------------------------------------------
% Send your comments and suggestions to <<< slzhou2021@163.com >>>                                  
% WARNING: Accuracy may not be guaranteed!!!!!  
% -------------------------------------------------------------------------
```
