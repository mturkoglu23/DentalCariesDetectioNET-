clc;
clear;

for k=1:340
b= '...\a (';
img=imread(strcat(b, num2str(k),').png'));

aa=size(img);
    if length(aa)==2
        img=cat(3,img,img,img);
    end
    
b2 = imsharpen(img,'Radius',3,'Amount',1);
b3=rgb2gray(b2);
% J = imadjust(b3);
J =imlocalbrighten(b3);
imgg=cat(3,J,J,J);

   imgg=imresize(imgg,[227 227]);

imwrite(imgg,fullfile(['....','\','a (',num2str(k),').png']));

end
