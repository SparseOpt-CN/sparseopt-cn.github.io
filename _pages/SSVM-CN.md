---
layout: archive
title: ""   
permalink: /SSVM-CN/
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


## 稀疏支持向量机
---
<div style="text-align:justify;">
经典的软阈值支持向量机（Support vector machine，SVM）模型为
</div>

\begin{equation}
\min_{(\mathbf{w};b)\in\mathbb{R}^{n+1}}~\frac{1}{2}\parallel \mathbf{w} \parallel^2 + C \sum_{i=1}^m\ell\left(1-y_i(b+ \mathbf{a}_i^\top\mathbf{w})\right) \tag{SVM}
\end{equation} 

<div style="text-align:justify;">
其中，样本矩阵 $\mathbf{A}=(\mathbf{a}_1$ $\ldots$ $\mathbf{a}_m)^\top$ $\in\mathbb{R}^{m\times n}$，标签向量 $\mathbf{y}=(y_1$ $\ldots$ $y_m)^\top$ $\in\mathbb{R}^{m}$，罚参数 $C>0$，$\ell$ 为损失函数。一种常用的损失函数是合页损失（hinge loss），其定义为 $\ell_{h}(t)=\max\{t,0\}$。下面考虑两种基于不同损失函数的 SVM 模型。 
</div>      

<p style="line-height: 2;"></p>

◻️ 阶梯函数正则 SVM 模型
\begin{equation}
\min_{(\mathbf{w};b)\in\mathbb{R}^{n+1}}~\frac{1}{2}\parallel \mathbf{w} \parallel^2 + C \sum_{i=1}^m\mathrm{step}\left(1-y_i(b+  \mathbf{a}_i^\top\mathbf{w})\right) \tag{L01SVM}
\end{equation} 
<div style="text-align:justify;">
其中，$\mathrm{step}$ 是阶梯（又称 0/1 损失）函数，定义为 $\mathrm{step}(t)=1$ 当 $t>0$，否则 $\mathrm{step}(t)=0$。若令零范数 $\parallel\mathbf{x}\parallel_0$ 表示 $\mathbf{x}$ 中非零元的个数和 $\mathbf{z}_+=(\max\{0,z_1\}$ $\ldots$ $\max\{0,z_m\})^\top$，则有 $\sum_{i=1}^m\mathrm{step}\left(1-y_i(b+  \mathbf{a}_i^\top\mathbf{w})\right)$ $=\| (1-\mathbf{A}\mathbf{w}-b\mathbf{y} )_+ \|_0$。 
</div>

◻️ 稀疏约束二次核 SVM 模型
\begin{equation}
\min_{\boldsymbol{\alpha}\in\mathbb{R}^{m}}~\frac{1}{2} \boldsymbol{\alpha}^\top \mathbf{Q} \boldsymbol{\alpha} + \sum_{i=1}^m h_{cC}(\alpha_i) -\mathbf{e}^\top\boldsymbol{\alpha}, ~~~~ \text{s.t.} ~~\mathbf{y}^\top\boldsymbol{\alpha}=0,~\parallel  \boldsymbol{\alpha} \parallel_0\leq s \tag{SCSVM}
\end{equation} 
<div style="text-align:justify;">
其中，核矩阵 $\mathbf{Q}\in\mathbb{R}^{m\times m}$ 的第 $(i,j)$ 个元素为 $Q_{ij}=y_iy_j\mathbf{a}_i^\top\mathbf{a}_j$，$\mathbf{e}=(1 \ldots 1)^\top$，正整数 $s\ll m$，参数 $C>c>0$；损失函数 $h_{cC}$ 定义为：$h_{cC}(t)=t^2/(2C)$ 当 $t>0$，否则 $h_{cC}(t)=t^2/(2c)$。实际上，若模型 (SCSVM) 中不含稀疏约束 $\parallel \boldsymbol{\alpha} \parallel_0\leq s$，则它是模型 (SCSVM) 在取 $\ell=\ell_{cC}$ 时的对偶问题，其中，$\ell_{cC}$ 定义为：$\ell_{cC}(t)=t^2/2$ 当 $t>0$，否则 $\ell_{cC}(t)=(c/C)t^2/2$。
</div>  

> <div style="text-align:justify;"> 
  根据表示定理（Representer Theorem），模型 (SVM) 的最优解 $\mathbf{w}$ 与对偶 SVM 的最优解 $\boldsymbol{\alpha}$ 满足 $\mathbf{w} = \sum_{i=1}^m \alpha_i y_i \mathbf{a}_i$。
  此时，对应于 $\boldsymbol{\alpha}$ 非零分量 $\alpha_i$ 的训练向量 $\mathbf{a}_i$ 被称为支持向量（support vectors）。因此，模型 (SFRSVM) 和 (SCSVM) 都能够减少支持向量的数量。 </div> 

---
<div style="text-align:justify;">
  程序包 - <a style="font-size: 16px; font-weight: bold;color:#006DB0" href="\files\SSVMpack-Matlab.zip" target="_blank">SSVMpack-Matlab</a>（点击下载）提供了 2 个求解器，其核心算法分别来自以下 2 篇文章，其中 $\texttt{NM01}$ 和 $\texttt{NSSVM}$ 分别用来求解模型 (SFRSVM) 和 (SCSNM)。
</div>  

> <div style="text-align:justify;"> <b style="font-size:14px;color:#777777">NM01</b> -<span style="font-size: 14px"> S Zhou, L Pan, N Xiu, and H Qi, Quadratic convergence of smoothing Newton's method for 0/1 loss optimization, SIOPT, 31:3184-3211, 2021. </span> </div>
> <div style="text-align:justify;">  <b style="font-size:14px;color:#777777">NSSVM</b> -<span style="font-size: 14px"> S Zhou, Sparse SVM for sufficient data reduction, IEEE TPAMI, 44:5560-5571, 2022. </span> </div>

---
<div style="text-align:justify;">
下面 Matlab 代码（Python 代码类似）展示了如何使用 $\texttt{SSVMpack}$ 求解稀疏 SVM（SSVM） 问题。用户需输入数据 ($\texttt{A}$，$\texttt{y}$)，必要时对数据进行归一化处理，再从 {'$\texttt{NM01}$'，'$\texttt{NSSVM}$'} 中选择一个作为求解器，然后运行求解。
</div>

<p style="line-height: 1;"></p>

```ruby
clc; close all; clear all; addpath(genpath(pwd));
  
load dhrb.mat;  
load dhrbclass.mat;  

[m0,n]    = size(A);         
A         = normalization(A,2); % data normalization 
m         = ceil(0.9*m0);       % data splition 
Train     = randperm(m0,m); 
Ttest     = setdiff(1:m0,Train); 
Atrain    = A(Train,:);     
Atest     = A(Ttest,:);
ytrain    = y(Train,:);     
ytest     = y(Ttest);    

t         = 1;
pars.C    = 0.25;
solver    = {'NM01','NSSVM'};
out       = SSVMpack(Atrain,ytrain,solver{t},pars);
acc       = accuracy(Atrain,out.w,ytrain);
tacc      = accuracy(Atest,out.w,ytest);

fprintf(' Training  Time:             %5.3fsec\n',out.time);
fprintf(' Training  Size:             %dx%d\n',size(Atrain,1),size(Atrain,2));
fprintf(' Training  Accuracy:         %5.2f%%\n', acc*100);
fprintf(' Testing   Size:             %dx%d\n',size(Atest,1),size(Atest,2));
fprintf(' Testing   Accuracy:         %5.2f%%\n',tacc*100);
fprintf(' Number of Support Vectors:  %d\n',out.sv); 
```
<div style="text-align:justify;">
Matlab 版程序包 $\texttt{SSVMpack}$ 的调用形式（Python 版的调用形式类似）如下所示。输入 ($\texttt{A}$，$\texttt{y}$，$\texttt{solver}$) 为必需项，其中 $\texttt{solver}$ 可从 {'$\texttt{NM01}$'，'$\texttt{NSSVM}$'} 中选择。参数 $\texttt{pars}$ 是可选的，但设置其中的一些参数（特别是 $\texttt{pars.C}$ 和 $\texttt{pars.s0}$）可以提升求解器的性能和解的质量。例如，如果 $\texttt{solver}$='$\texttt{NSSVM}$'，则设置合适的 $\texttt{pars.s0}$ 可提升解的质量。另一个重要参数是 $\texttt{pars.C}$，可通过交叉验证（Cross-validation）进行调优。</div>
<p style="line-height: 1;"></p>

```ruby
function out = SSVMpack(A,y,solver,pars)
% -------------------------------------------------------------------------
% This package aims to solve the binary classification problems
% Inputs:
%  A:       The smaple matrix \in R^{m-by-n},                    (REQUIRED)
%  y:       The binary label \in R^m, b_i\in{-1,1}               (REQUIRED)    
%  solver:  A text string, can be one of {'NM01','NSSVM'}        (REQUIRED)            
%  pars:    Parameters are optional                              (OPTIONAL) 
%           -------------  For NSSVM --------------------------------------
%           pars.alpha --  Starting point in \R^m       (default zeros(m,1))
%           pars.s0    --  The initial sparsity    (default n(log(m/n))^2))
%           pars.C     --  A positive scalar in (0,1]        (default  1/4)  
%           pars.c     --  A positive scalar in (0,1]        (default 1/40)  
%           pars.tune  --  Tune the sparsity level              
%                          Do not tune the sparsity level       (default 0)
%           pars.maxit --  Maximum number of iterations      (default 1000) 
%           pars.tol   --  Tolerance of the halting criteria (default 1e-4) 
%           pars.disp  --  Display results for each step        (default 1)  
%                          Do not display results for each step 
%           -------------  NM01 -------------------------------------------
%           pars.x0    --  The initial point           (default zeros(n,1))
%           pars.C     --  The penalty parameter                (default 1)
%           pars.tau   --  A useful paramter                    (default 5)
%           pars.maxit --  Maximum number of iterations      (default 1000)  
%           pars.tol   --  Tolerance of the halting criteria (default 1e-4) 
%           pars.disp  --  Display results for each step        (default 1)  
%                          Do not display results for each step 
% -------------------------------------------------------------------------
% Outputs:
%     out.w:      The solution of the primal problem, i.e., the classifier
%     out.sv:     Number of support vectors 
%     out.time    CPU time
%     out.iter:   Number of iterations
%     Out.acc:    Classification accuracy
% -------------------------------------------------------------------------
% Send your comments and suggestions to <slzhou2021@163.com> 
% Warning: Accuracy may not be guaranteed !!!!!! 
% -------------------------------------------------------------------------
```
