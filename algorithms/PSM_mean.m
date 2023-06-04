function mu = PSM_mean(As)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
% Implementation based on the paper:
% @article{BONNABEL20133202,
% author = "Silvère Bonnabel and Anne Collard and Rodolphe Sepulchre",
% title = "Rank-preserving geometric means of positive semi-definite matrices",
% journal = "Linear Algebra and its Applications",
% volume = "438",
% number = "8",
% pages = "3202 - 3216",
% year = "2013",
% issn = "0024-3795",
% doi = "https://doi.org/10.1016/j.laa.2012.12.009",
% url = "http://www.sciencedirect.com/science/article/pii/S0024379512008646",
% keywords = "Matrix means, Geometric mean, Positive semi-definite matrices, Riemannian geometry, Symmetries, Singular value decomposition, Principal angles",
% abstract = "The generalization of the geometric mean of positive scalars to positive definite matrices has attracted considerable attention since the seminal work of Ando. The paper generalizes this framework of matrix means by proposing the definition of a rank-preserving mean for two or an arbitrary number of positive semi-definite matrices of fixed rank. The proposed mean is shown to be geometric in that it satisfies all the expected properties of a rank-preserving geometric mean. The work is motivated by operations on low-rank approximations of positive definite matrices in high-dimensional spaces."
% }
% 
% Karcher mean implementation from
% https://www.cs.colostate.edu/~vision/summet/      % TODO get corresponding citations
% Ando mean implementation from 
% http://bezout.dm.unipi.it/software/mmtoolbox/     % TODO get corresponding citations
% 
% ---------------------------------------------------------------------
%
% Copyright (c) 2020 Kerstin Bunte
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
% 
%    1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
%  
%    2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the
%       documentation and/or other materials provided with the distribution.
%  
%    3. Neither name of copyright holders nor the names of its contributors may be used to endorse or promote products derived from this 
%       software without specific prior written permission.
%  
%  
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
% LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
% CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
% PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
% LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%
% ---------------------------------------------------------------------
N = size(As,3);

ranks = arrayfun(@(j) rank(As(:,:,j)), 1:N);
m = unique(ranks);
if (length(unique(ranks))>1)
	error('X and Y must be of the same rank');
end

Asizes = size(As);
n = unique(Asizes(1:2));
if (length(m)>1)
	error('As must be of the same dimensions and square');
end

Ds = cell2mat(arrayfun(@(j) eig(As(:,:,j)), 1:N , 'uni', 0));
Ds(abs(Ds)<10e-10) = 0;
noNegEigs=arrayfun(@(j) isempty(Ds(Ds(:,j)<0,j)), 1:N);
if (sum(noNegEigs)~=length(noNegEigs))
	error('Input matrices X and Y must be positive semi definite');
end

if N==2
    % this is the implementation of the analytic solution for 2 matrices from Bonnabel himself
    alpha = 1/2;
    Us = zeros(n,m,N);Vs = zeros(n,m,N);
    sqRs = zeros(m,m,N);
    for j=1:N
        [u,l,v] = svd(As(:,:,j),0);
        Us(:,:,j) = u(:,1:m);
        sqRs(:,:,j) = l(1:m,1:m);
        Vs(:,:,j) = v(:,1:m);
    end
    % (U1*O1)'*(U1*O1)  all bases correspond to the same p-dim subspace U_i'U_i
    % equivalence classes U_iO(p) are called the fibers
    [O1, cosSigma, O2] = svd(Us(:,:,1)'*Us(:,:,2),0);
    
    my_angles = acos(diag(cosSigma));
    my_Sigma = diag(my_angles);    
    if(real(sum(my_angles))>pi/2)
        %angle_idx = find(my_angles>pi/2);
        %test = sprintf('%i (%f>pi/2), ', [angle_idx;my_angles(angle_idx)] );
        warning('principal angles between span(U1) and span(U2) should be smaller than pi/2');
    end    
%     Us(:,:,1)'*Us(:,:,2) - O1*diag(cos(my_angles))*O2'
%     Us(:,:,1)'*Us(:,:,2) - O1*funm(my_Sigma,@cos)*O2'
    Y1 = Us(:,:,1)*O1; % principal vectors for X
    Y2 = Us(:,:,2)*O2; % principal vectors for Y
    
    sqR1 = Y1'*As(:,:,1)*Y1; % R^2 = Y'*A*Y representatives
    sqR2 = Y2'*As(:,:,2)*Y2;
    
%     Mean A1 and A2 = W*(R_1^2#R_2^2)*W'
%     R_1^2#R_2^2 is the Ando mean of R_1^2 and R_2^2:    
    SRoot_sqR1 = sqrtm(sqR1);
%     inv_SRoot_sqR1 = inv(SRoot_sqR1);
%     AndoMean = SRoot_sqR1* sqrtm( inv_SRoot_sqR1*sqR2*inv_SRoot_sqR1 ) * SRoot_sqR1;
%     AndoMean = SRoot_sqR1* sqrtm( (SRoot_sqR1\sqR2)/SRoot_sqR1 ) * SRoot_sqR1;
    if (alpha == 0.5)
%         AndoMean = SRoot_sqR1* sqrtm( inv(SRoot_sqR1)*sqR2*inv(SRoot_sqR1) ) * SRoot_sqR1;
        AndoMean = SRoot_sqR1* sqrtm( (SRoot_sqR1\sqR2)/SRoot_sqR1 ) * SRoot_sqR1; % Rz = sqRx*sqrtm(isqRx*Ry*isqRx)*sqRx;
    else
        AndoMean = SRoot_sqR1* mpower((SRoot_sqR1\sqR2)/SRoot_sqR1 ,alpha) * SRoot_sqR1; % Rz = sqRx*funm(isqRx*Ry*isqRx,@expmat,[],alpha)*sqRx;
    end
    
%     W is Riemannian mean of Y1 and Y2 from Bonnabel implementation based on a different X than described in the paper
%     X = (eye(n) - Y1*Y1')*Y2*pinv(diag(my_angles)); % 
%     W = Y1*funm(my_Sigma*alpha,@cos) + X*funm(my_Sigma*alpha,@sin);    
%     mu = W*AndoMean*W';
 
%     own implementation directly from the paper
    myX    = (Y2-Y1 * funm(my_Sigma,@cos)) * pinv(funm(my_Sigma,@sin));
    mean_Y =  Y1 * funm(my_Sigma*alpha,@cos) + myX * funm(my_Sigma * alpha,@sin);
    mu = mean_Y*AndoMean*mean_Y'; % this agrees with the approximation implementation for arbitrary matrices though, so I assume this is correct
else
    % STEP 1:
    % U_i is any orthonormal basis of the span of A_i
    Uis = cell(N,1);Vis = cell(N,1);sqRis = cell(N,1);
    for j=1:N
        [u,l,v] = svd(As(:,:,j),0);
        Uis{j} = u(:,1:m);
        sqRis{j} = l(1:m,1:m);
        Vis{j} = v(:,1:m);
    end
    clear u l v;
    % TODO: CHECK IF SUBSPACES SPANNED BY COLUMNS OF A_i's ARE ENCLOSED IN A GEODESIC BALL OF RADIUS LESS THAN pi/(4*sqrt(2)) IN Gr(p,n)! 
    % max distance of points pi/(2*sqrt(2))
    % this could otherwise lead to problems if several local minima exist!
%     maxDist = pi/(2*sqrt(2));
%     GrDist(NewMU, MU, size(MU,2), 'DGEO')

%     1. subspaces spanned by columns of the A_i's are enclosed in a geodesic ball of radius less than pi/(4*sqrt(2)) in Gr(p,n) 
%     Grassmann manifold Gr(p,n) is set of p-dim subspaces of R^n, represented by equivalence classes St(p,n)/O(p)
%     St(p,n)=O(n)/O(n-p) is the Stiefel manifold (set of nxp matrices with orthonormal columns: U^TU=I_p    
%     define W element of St(p,n) as orthonormal basis of the unique Karcher mean of the U_iU_i^T
% Karcher mean exists uniquely in geodesic balls with sufficiently small radius
% Karcher mean of projectors in Gr(p,n) is natural rank-preserving rotation invariant mean that is well-defined on a subset of the boundary of the cone.
% words, the injectivity radius at any point, i.e. roughly speaking the distance at which the geodesics
% The Karcher mean of N subspaces S1,...,SN of Gr(p, n) is defined as the least squares solution that minimizes X → sum_1^N d_Gr(p,n)(X,S_i)^2. 
% The latter function is equal to \sum_i=1^N\sum_j^N θ_ij where θij^2 where θij is the jth principal angle between X and S_i.
% [2] B. Afsari, Riemannian Lp center of mass: existence, uniqueness, and convexity, Proc. Amer. Math. Soc. 139 (2) (2011) 655–673.
    % STEP 2:
    % TODO: if this is too slow one can use the faster approximate "flag mean"
    if exist('KarcherMean','file')~=2 error('This implementation uses the Karcher Mean implementation from summet. Please download it from https://www.cs.colostate.edu/~vision/summet/ and include in the path.'); end
    subspace_KarcherMu = KarcherMean(Uis, Uis{randi(N,1)}, 10^-5, 1000,1);                     % use of summet toolbox    
    
    % check the grassman geodesic distance to the computed mean:
%   d_Grs2Mu = arrayfun(@(j) GrDist(Uis{j},subspace_KarcherMu, size(Uis{j},2), 'DGEO'),1:N)
%   Gr_manifold = grassmannfactory(size(subspace_KarcherMu,1),size(subspace_KarcherMu,2));    % use of manopt toolbox
%   arrayfun(@(j) Gr_manifold.dist(Uis{j},subspace_KarcherMu), 1:N)

    % THIS IS PROB WRONG IN PAPER OR WEIRDLY WRITTEN FOR THE COMPUTATION OF THE KARCHER MEAN: 
    % define W \in St(p,n) as orthonormal basis of the unique Karcher mean of the U_iU_i^T's
%     UUT = cellfun(@(u) u*u', Us,'uni',0);KarcherMean(UUT, UUT{randi(N,1)}, 10^-3, 100,1)
    nW = subspace_KarcherMu;    % cat(3, W, nW) % to test if 2 matrices give the same as the analytic solution
    % 2. For each i, compute 2 bases Y_i and W_i of span(U_i) and span(W),
    % such that d_St(p,n)(Y_i,W_i)=d_Gr(p,n)(span(U_i),span(W)), i.e. solve problem (6):
    %   (Q_1,Q_2) = argmin_{(O_1,O_2)\inO(p)xO(p)} d_St(p,n)(U_1 O_1, U_2 O_2)
    % Let S_i^2 = Y_i^t A_i Y_i. The ellipsoid A_i rotated to the mean subspace writes W_i S_i^2 W_i^t.  
    % STEP 3:
    % for i=1:N:
%         - SVD of U_i^t W yields 2 orthogonal matrices O_i, O_i^W such that O_i^t U_i^t W O_i^W is a diagonal matrix
%         - Let Y_i=U_i O_i and W_i=W O_i^W. Let S_i^2=Y_i^t A_i Y_i. Let T_i^2=W^t W_i S_i^2 W_i^t W
    Ti2s = cell(N,1);
%     Yis = cell(N,1);Wis = cell(N,1);Si2s = cell(N,1);
    for j = 1:N
        [O,~,OW] = svd(Uis{j}'*nW,0);
%         O'*Uis{j}'*nW*OW % this is a diagonal matrix indeed
%         Yis{j} = Uis{j}*O;Wis{j} = nW*OW;Si2s{j}= Yis{j}'*As(:,:,j)*Yis{j};
%         Ti2s{j}= nW'*Wis{j}*Si2s{j}*Wis{j}'*nW;
        Yj = Uis{j}*O;              
%         if j==1 cat(3,Y1,Yj), elseif j==2 cat(3,Y2,Yj), end           % to test if 2 matrices give the same as the analytic solution
        Wj = nW*OW;
        Sj2 = Yj'*As(:,:,j)*Yj;
%         if j==1 cat(3,sqR1,Sj2), elseif j==2 cat(3,sqR2,Sj2), end     % to test if 2 matrices give the same as the analytic solution
        Ti2s{j}= nW'*Wj*Sj2*Wj'*nW;        
%         [eigs(Wj*Sj2*Wj'),eigs(As(:,:,j))]
    end
    clear Yj Wj Sj2 O OW;
    % STEP 4: 
    % Compute the geometric mean in the low-rank cone P_n: M(T_1^2,..., T_k^2) using methods in the literature [3,5,11].
    % [3] T. Ando, C.K. Li, R. Mathias, Geometric means, Linear Algebra Appl. 385 (2004) 305–334.
    % [5] M. Arnaudon, C. Dombry, A. Phan, Le Yang, Stochastic algorithms for computing means of probability measures, Stochastic Process. Appl. 122 (2012) 1437–1455.
    % [11] D. Bini, B. Meini, F. Poloni, An effective matrix geometric mean satisfying the Ando–Li–Mathias properties, Math. Comp. 79 (2010) 437–452.
%     AndoMean_Ts = alm(Ti2s{:}); % implementation from mmtoolbox, from Ando 2004
    AndoMean_Ts = cheap(Ti2s{:}); % implementation from mmtoolbox, from Bini 2010
    % STEP 5:
    % The geometric mean is: WM(T_1^2,..., T_k^2)W^t
    mu = nW*AndoMean_Ts*nW';
end
end

