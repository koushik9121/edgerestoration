% This program removes noise from an image using linear second order
% diffusion filter u_t=u_xx+u_yy with homogeneous neumann boundary conditions
clc
clear all
a = imread('monalisa.png');




%a = rgb2gray(RGB);




size(a)
ref=im2double(a);
%ref=rgb2gray(ref);
%noisy=imnoise(ref,'gaussian',0.01);
g=@(s,lam)(exp((-(s)^5)/(5*(lam)^5)));
%noisy=imnoise(ref,'salt & pepper',0.2);
%noisy=gamrnd(ref,1./ref,size(ref));
U=(ref);
verbose=2;
if verbose
figure(verbose);
subplot(1,2,1);
imshow(U);
title('Noisy image')
drawnow;
end
nitr=80;
dt=0.1;
r=dt;
sigma=3;
K=2;
%{
    for n=1:nitr
        Umc=translateImage(U,-1,0); % Shifts down, ie, In place of U(i,j) we will have U(i-1,j)
        Upc=translateImage(U,1,0); % Shifts up, ie, In place of U(i,j) we will have   U(i+1,j)
        Ucm=translateImage(U,0,-1);% Shifts right, ie, In place of U(i,j) we will have U(i,j-1)
        Ucp=translateImage(U,0,1); % Shifts left, ie, In place of U(i,j) we will have U(i,j+1)
        
        U=U+ K*r*(Upc+Umc+Ucp+Ucm-4*U); % updating U
        
        if verbose
            figure(verbose);
            subplot(1,2,2);
            imshow(U);
            title(n)
            drawnow;
            end
            pause(2) % pause the display for two seconds
                
                end
                %}
for n=1:nitr   % Evolving upto 't'
U=real(U);
Umc=translateImage(U,-1,0); % Shifts down, ie, In place of U(i,j) we will have U(i-1,j)
Upc=translateImage(U,1,0); % Shifts up, ie, In place of U(i,j) we will have   U(i+1,j)
Ucm=translateImage(U,0,-1);% Shifts right, ie, In place of U(i,j) we will have U(i,j-1)
Ucp=translateImage(U,0,1); % Shifts left, ie, In place of U(i,j) we will have U(i,j+1)
gauss_filter = imgaussfilt(U,sigma);
gmc=translateImage(gauss_filter,-1,0); % Shifts down, ie, In place of U(i,j) we will have U(i-1,j)
gpc=translateImage(gauss_filter,1,0); % Shifts up, ie, In place of U(i,j) we will have   U(i+1,j)
gcm=translateImage(gauss_filter,0,-1);% Shifts right, ie, In place of U(i,j) we will have U(i,j-1)
gcp=translateImage(gauss_filter,0,1);
grad_U=r*sqrt(0.25*((Upc-Umc).^2)+0.25*((Ucp-Ucm).^2));
grad_U_sigma=r*sqrt(0.25*((gpc-gmc).^2)+0.25*((gcp-gcm).^2));
%D=eye(size(U)); %linear diffusion
%mod1=abs(grad_U_sigma); %non linear isotropic diffusion
%req=sqrt(grad_U_sigma*transpose(grad_U_sigma)); %non linear anisotropic
%G=g(D,3.5);
[V,dia] = eig(U);
inner=(grad_U*V);
inner=abs(inner);
G=V*transpose(V)*g(inner,4);
D=G;
Dmc=translateImage(D,-1,0); % Shifts down, ie, In place of U(i,j) we will have U(i-1,j)
Dpc=translateImage(D,1,0); % Shifts up, ie, In place of U(i,j) we will have   U(i+1,j)
Dcm=translateImage(D,0,-1);% Shifts right, ie, In place of U(i,j) we will have U(i,j-1)
Dcp=translateImage(D,0,1); % Shifts left, ie, In place of U(i,j) we will have U(i,j+1)
X=U+K*r*(((Dpc+D)/2).*(Upc-U)-(((Dmc+D)/2).*(U-Umc)));
Y=K*r*(((Dcp+D)/2).*(Ucp-U)-(((Dcm+D)/2).*(U-Ucm)));
U=(X+Y); % updating U
U=real(U);
if verbose
figure(verbose);
subplot(1,2,2);
imshow(U);
title(n)
drawnow;
end
pause(2) % pause the display for two seconds

end

figure(6)
imshow(U) % Filtered image at given 't'

