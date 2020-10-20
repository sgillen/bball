% load itall x_impacts
% x4 = x_impacts';
%load bball_dim.mat
% x4 = state_mat(2:end,2:end);
x4 = state_mat;
x_axis = 1:1:size(x4,1);
plot(x_axis, x4(:, 10))
figure
plot(x_axis, x4(:, 5))

%load itall_2 x_impacts
%x4 = [x4; x_impacts'];
%load itall_big x_impacts
%x4 = [x4; x_impacts'];

xmin = min(x4);
xmax = max(x4);
sf = 1./(xmax-xmin);
% for n=1:3
%     x4(:,n) = sf(n)*x4(:,n);
% end

for n=1:size(x4,2)
    x4(:,n) = sf(n)*x4(:,n);
end

dlist = 10.^[2:-.25:-4];
nlist = 0*dlist; % to be determined, below
for ndo = 1:length(dlist)   
    Mid = zeros(1,length(x4(:,1)));
    mi = 1 ;
    Mid(mi) = 1;
    bFilled = 0*x4(:,1);
    bFilled(1) = true; % boolean, tracking which points are inside "sphere
    % radius" of another mesh point (possibly itself) already
    %dth = .01; % meshing threshold value
    dth = dlist(ndo);
    % Fill up the mesh:
    fi = find(~bFilled);
    while ~isempty(fi) %sum(~bFilled)~=0 % still have points to add to mesh
        N = length(fi);
        dx_fi = ones(N,1)*x4(Mid(mi),:) - x4(fi,:);
        fi2 = find(sqrt(sum(dx_fi.^2,2)) < dth);
        bFilled(fi(fi2)) = true;
        
        
        fi = find(~bFilled);
        if ~isempty(fi)
            mi=mi+1;
            Mid(mi) = fi(1); % next available mesh point
        end
    end
    nlist(ndo) = mi
end

use_id = [1:6]+7; % only fit a line to first n points, in dlist....
pp = polyfit(log(dlist(use_id)),log(nlist(use_id)),1)
frac_dim = -pp(1)

figure(55); clf
loglog(dlist,nlist,'ro'); hold on
xfit = [dlist(1), dlist(end)];
yfit = exp(polyval(pp,log(xfit)));
loglog(xfit,yfit,'b--'); grid on
loglog(xfit,0*yfit+length(x4(:,1)),'b--')
loglog(xfit,0*yfit+1,'b--')
loglog(dlist(use_id),nlist(use_id),'ro','MarkerFaceColor','k'); 

xlabel('d_{thr}, threshold radius')
ylabel('N, number of mesh points')