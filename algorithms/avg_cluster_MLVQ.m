function [allCLSmodels,ZL,D_L] = avg_cluster_MLVQ(useModels, clusters)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    Lambdas = cellfun(@(x) x.A'*x.A, useModels(:),'uni',0);
    Lambdas = cat(3,Lambdas{:});%     arrayfun(@(x) trace(Lambdas(:,:,x)),1:size(Lambdas,3))
    N = size(Lambdas,3);
    m = min(arrayfun(@(x) rank(Lambdas(:,:,x)),1:size(Lambdas,3)));%rank(Lambdas(:,:,1));    
%     Ws = cellfun(@(x) x.w, models(actFold,:),'uni',0);
%     Ws = cat(3,Ws{:});
%     sphere_dist = @(X,Y) acos( diag((X./arrayfun(@(j) norm(X(j,:)),1:size(X,1) )') * (Y'./arrayfun(@(j) norm(Y(j,:)),1:size(Y,1) ))) );
%     D_W = nan(1,N*(N-1)*0.5); 
%     k = 1;
%     for i = 1:N-1
%         D_W(k:(k+N-i-1)) = arrayfun(@(m2) mean(sphere_dist(Ws(:,:,i),Ws(:,:,m2))) ,i+1:N );
%         k = k + (N-i);
%     end
%     figure;Z = linkage(D_W,'ward');[H, T] = dendrogram(Z);
%     cluster_mb = cluster(Z,'Cutoff',pi/4,'Criterion','distance')' %cluster(z,'maxclust',5)
    Uis = cell(N,1);Vis = cell(N,1);sqRis = cell(N,1);
    for j=1:N
        [u,l,v] = svd(Lambdas(:,:,j),0);
        Uis{j} = u(:,1:m);
        sqRis{j} = l(1:m,1:m);
        Vis{j} = v(:,1:m);
    end
    clear u l v;
    manifold = grassmannfactory(size(Uis{1},1),m);
    D_L = nan(1,N*(N-1)*0.5); 
    k = 1;
    for i = 1:N-1
        D_L(k:(k+N-i-1)) = arrayfun(@(m2) manifold.dist(Uis{i},Uis{m2}) ,i+1:N );
        k = k + (N-i);
    end
% pi/(4*sqrt(2))
%     figure;
    ZL = linkage(D_L,'ward');
%     [HL, TL] = dendrogram(ZL);
%        'single'    --- nearest distance (default)
%        'complete'  --- furthest distance
%        'average'   --- unweighted average distance (UPGMA) (also known as group average)
%        'weighted'  --- weighted average distance (WPGMA)
%        'centroid'  --- unweighted center of mass distance (UPGMC)
%        'median'    --- weighted center of mass distance (WPGMC)
%        'ward'      --- inner squared distance (min variance algorithm)   
%     cluster_mb_L = cluster(ZL,'Cutoff',pi,'Criterion','distance')' 
%     cluster_mb_L = cluster(ZL,'maxclust',length(unique(cluster_mb)))';
    allCLSmodels = cell(1,length(clusters));
    for j = 1:length(clusters)
        actCLSNumber = clusters(j);        
        useClustering = cluster(ZL,'maxclust',actCLSNumber)';        
        actClsNb = length(unique(useClustering));
        ClsModels = cell(1,actClsNb);
        for actC = 1:actClsNb
            actCIdx = find(useClustering==actC);
            if length(actCIdx)==1
                actModel = useModels{actCIdx};
            else
                actModel = avg_ensemble_ALVQ(useModels(actCIdx));
            end
%            actModel.beta = actModel.theta; % my classify file asks for beta not theta as a field. Maybe some unpushed changes in angleLVQtoolbox?
            actModel.A = actModel.A./sqrt(trace(actModel.A'*actModel.A));
            actModel.w = cell2mat(arrayfun(@(x) actModel.w(x,:)./norm(actModel.w(x,:)),1:size(actModel.w,1),'uni',0)'); % normalize to hypersphere
            actModel.clsIdx = actCIdx;
            ClsModels{actC} = actModel;
        end
        allCLSmodels{j} = ClsModels;
    end
    
end

