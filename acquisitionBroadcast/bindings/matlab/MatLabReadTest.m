%Written by Iasonas Oikonomidis , 2013
%urlread converts everything to UTF8 <- check that it doesnt do that ( see urlread.m )!
%   changing  output = native2unicode(typecast(byteArrayOutputStream.toByteArray','uint8'),'UTF-8'); to 
%             output = typecast(byteArrayOutputStream.toByteArray','uint8');
%$MATLAB_HOME$/toolbox/matlab/iofun

tmpin = urlread('http://139.91.185.11:8080/rgb.raw');
%%
tmp = reshape(tmpin, [3 640 480]);
tmp = permute(tmp, [3 2 1]);
figure
imagesc(single(tmp) / 255)
axis equal
axis off
%%

tmpin = urlread('http://139.91.185.11:8080/depth.raw');

%%

masks = ones(size(tmpin));
masks(1:2:length(masks)) = 255;
tmp = single(tmpin) .* masks;
tmp = tmp(1:2:end) + tmp(2:2:end);
tmp = reshape(tmp, [640 480]);
%tmp = permute(tmp, [3 2 1]);
imagesc(tmp' / 255)
axis equal
axis off

%%
tmp = urlread('http://139.91.185.11:8080/control.html?play=0');
tmp = urlread('http://139.91.185.11:8080/control.html?snap=0');
tmp = urlread('http://139.91.185.11:8080/control.html?seek=0');
