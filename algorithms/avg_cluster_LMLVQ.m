function [allCLSmodels,ZL,D_L] = avg_cluster_LMLVQ(useModels, clusters)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    if(~exist('grassmannfactory.m','file'))
        error('the grassmannfactory of manopt is missing. Download it at manopt.org and include it in the path.');
    end
    nb_L = length(useModels{1}.A);
    N = length(useModels);
    
    for j=1:N
        useModels{j}.A = cellfun(@(x) x./sqrt(trace(x'*x)),useModels{j}.A,'uni',0);
    end
    
    D_Ls = nan(nb_L,N*(N-1)*0.5);
    for actClass = 1:nb_L
        Lambdas = cellfun(@(x) x.A{actClass}'*x.A{actClass}, useModels(:),'uni',0);
        Lambdas = cat(3,Lambdas{:});
        %     arrayfun(@(x) trace(Lambdas(:,:,x)),1:size(Lambdas,3))
        m = min(arrayfun(@(x) rank(Lambdas(:,:,x)),1:size(Lambdas,3)));%rank(Lambdas(:,:,1));    

        Uis = cell(N,1);Vis = cell(N,1);sqRis = cell(N,1);
        for j=1:N
            [u,l,v] = svd(Lambdas(:,:,j),0);
            Uis{j} = u(:,1:m);
            sqRis{j} = l(1:m,1:m);
            Vis{j} = v(:,1:m);
        end
        clear u l v;
        manifold = grassmannfactory(size(Uis{1},1),m);
        k = 1;
        for i = 1:N-1
            D_Ls(actClass,k:(k+N-i-1)) = arrayfun(@(m2) manifold.dist(Uis{i},Uis{m2}) ,i+1:N );
            k = k + (N-i);
        end
    end
% pi/(4*sqrt(2))
    D_L = mean(D_Ls);
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
                actModel = avg_ensemble_LALVQ(useModels(actCIdx));
            end
%             actModel.beta = actModel.theta; % my classify file asks for beta not theta as a field. Maybe some unpushed changes in angleLVQtoolbox?
%             actModel.A = actModel.A./sqrt(trace(actModel.A'*actModel.A));
%             actModel.w = cell2mat(arrayfun(@(x) actModel.w(x,:)./norm(actModel.w(x,:)),1:size(actModel.w,1),'uni',0)'); % normalize to hypersphere
            actModel.clsIdx = actCIdx;
            ClsModels{actC} = actModel;
        end
        allCLSmodels{j} = ClsModels;
    end
    
end

