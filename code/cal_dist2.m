function [dmat] = cal_dist2(sketch,photo,var)

for i = 1:size(sketch,1)
    for k = 1:size(photo,1)
        temp = 0;
        temp = (sketch(i,:) - photo(k,:)).^2./(var(i,:).^2);
        dmat(i,k) = sum(temp);
    end
end