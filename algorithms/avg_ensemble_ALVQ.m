function avg_AModel = avg_ensemble_ALVQ(Models, dimension)
%avg_ensempble_ALVQ: Summary of this function goes here
%   Models: constitutes a cell array of ALVQ model structs, typically with fields beta, c_w, w and A

allFields = cellfun(@(x) fieldnames(x),Models,'uniform',0);
if(sum(cellfun(@(x) isequal(allFields{1},x),allFields(2:end))) < length(allFields)-1)
     error('Not all models have the same fields!');
end

% fieldStr = sprintf("%s",arrayfun(@(x) sprintf("'%s',[],",allFields{1}{x}),1:length(allFields{1}),'uni',1));
% avg_AModel = eval(sprintf("struct(%s)",fieldStr{1}(1:end-1)));
fieldStr = sprintf('''%s'',[],',allFields{1}{:});
avg_AModel = eval(sprintf('struct(%s)',fieldStr(1:end-1)));
for act = 1:length(allFields{1})
    actField = allFields{1}{act};
    switch(actField)
        case('c_w')
            % TODO check here for equivalent class labeling of the prototypes
            values = mean(cell2mat(cellfun(@(x) eval(sprintf('x.%s',actField)),Models,'uni',0)),2);
        case('w')
            firstEntry = eval(sprintf('Models{1}.%s',actField));
            concated = cell2mat(cellfun(@(x) eval(sprintf('x.%s',actField)),Models,'uni',0));
            allws = reshape(concated,size(firstEntry,1),size(firstEntry,2),size(concated,2)/size(firstEntry,2));
%             allws = cell2mat(cellfun(@(x) eval(sprintf('x.%s',actField)),Models,'uni',0));
            euclid_mn = mean(allws,3);
            sphere_dist = @(sphereDat,y) sum( acos( sphereDat*y'/norm(y) ) );            
            values = nan(size(euclid_mn));
            for actw=1:size(euclid_mn,1)
                actWs = squeeze(allws(actw,:,:))';
                sphere_dat = actWs./arrayfun(@(j) norm(actWs(j,:)),1:size(actWs,1))';
                options = struct( 'Display','iter', 'GradConstr',0, 'GoalsExactAchieve',1, 'TolFun',1e-6, ...
                                  'MaxIter',2500, 'MaxFunEvals', 1000000, 'TolX',1e-10, 'DiffMinChange',1e-10, 'HessUpdate','lbfgs' );
                variable = fminlbfgs(@(variable) sphere_dist(sphere_dat,variable), euclid_mn(actw,:),options);
                fprintf('Distance of euclidean mean to all points on sphere: %.3f\n',sphere_dist( sphere_dat,euclid_mn(actw,:) )); 
                fprintf('Distance of sphere mean to all points on sphere: %.3f\n',sphere_dist(sphere_dat,variable)); 
%                 [variable,fval] = fminsearch(@(variable) sphere_dist(actWs./arrayfun(@(j) norm(actWs(j,:)),1:size(actWs,1))',variable),euclid_mn(actw,:));
                values(actw,:) = variable;
            end
%             test = reshape(cell2mat(cellfun(@(x) eval(sprintf('x.%s',actField)),Models,'uni',0)),7,496,5)
%             test(:,:,2)-Models{2}.w
        case('A')
            % TODO: in PSM_mean:
            % TODO: CHECK IF SUBSPACES SPANNED BY COLUMNS OF A_i's ARE ENCLOSED IN A GEODESIC BALL OF RADIUS LESS THAN pi/(4*sqrt(2)) IN Gr(p,n)!
            % This could otherwise lead to problems if several local minima exist!            
            % TODO    test = newA/trace(newA'*newA); % fixing of the trace needs to be checked for PSD avg
            concated = cellfun(@(x) eval(sprintf('x.%s''*x.%s',actField,actField)),Models,'uni',0);
            Lambdas = cat(3,concated{:});
            mnL = PSM_mean(Lambdas); % compute the rank preserving rotation energy preserving geodesic mean to replace the naive avg
            
            [u,s,~] = svd( mnL ,0);
            if ~exist('dimension','var'), dimension = length(find(diag(s)>10e-10)); end
            values = sqrt(s(1:dimension,1:dimension))*u(:,1:dimension)';
            
            %  This was the naive implementation
%             firstEntry = eval(sprintf('Models{1}.%s',actField));
%             concated = cell2mat(cellfun(@(x) eval(sprintf('x.%s',actField)),Models,'uni',0));
%             allAs = reshape(concated,size(firstEntry,1),size(firstEntry,2),size(concated,2)/size(firstEntry,2));
%             mnLambda = allAs(:,:,1)'*allAs(:,:,1);
%             for actmat = 2:size(allAs,3)
%                 mnLambda = mnLambda + allAs(:,:,actmat)'*allAs(:,:,actmat);
%             end
%             mnLambda = mnLambda./size(allAs,3);
%                       
%             [V,D] = eig(mnLambda,'vector'); % [U,S,V] = svd(actLambda);
%             [~,idx] = sort(D,'descend');
%             D = D(idx);
%             V = V(:,idx);
%             if ~exist('dimension','var'), dimension = length(D(D>0)); end
%             values = bsxfun(@times, sqrt(D(1:dimension))',V(:,1:dimension))';
        case('beta')
             values = mean(cellfun(@(x) eval(sprintf('x.%s',actField)),Models));     
       case('theta')
             values = mean(cellfun(@(x) eval(sprintf('x.%s',actField)),Models));
       otherwise 
%              values = (cell2mat(cellfun(@(x) eval(sprintf('x.%s',actField)),useModels,'UniformOutput',0))');
    end
    eval(sprintf('avg_AModel.%s=values;',actField));
end
end
