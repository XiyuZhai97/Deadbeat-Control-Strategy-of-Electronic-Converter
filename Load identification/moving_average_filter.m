function output_f=moving_average_filter(input,window_length)
%滑动平均滤波  对电感电流进行滤波，滤除开关纹波
k = 0;
m = 0;
output_f=zeros(length(input),1);
for i = 1:length(input)
    m = m+1;
    if i+window_length-1 > length(input)
        break
    else
        for j = i:window_length+i-1
            k = k+1;
            W(k) = input(j) ;
        end
        output_f(m) = mean(W);
        k = 0;
    end
end
