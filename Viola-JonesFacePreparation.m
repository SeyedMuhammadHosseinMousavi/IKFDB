clear;
%Detect objects using Viola-Jones Algorithm
%To detect Face
FDetect = vision.CascadeObjectDetector('FrontalFaceCART','MergeThreshold',16');
% NoseDetect = vision.CascadeObjectDetector('Nose','MergeThreshold',16);
% MouthDetect = vision.CascadeObjectDetector('Mouth','MergeThreshold',16);
% EyePairBigDetect = vision.CascadeObjectDetector('EyePairBig','MergeThreshold',16);
%Read the input images
% 'path' is the folder name containing images
path='Sadness';
fileinfo = dir(fullfile(path,'*.jpg'));
filesnumber=size(fileinfo);
for i = 1 : filesnumber(1,1)
images{i} = imread(fullfile(path,fileinfo(i).name));
    disp(['Loading image No :   ' num2str(i) ]);
end;
%Returns Bounding Box values based on number of objects
for i = 1 : filesnumber(1,1)
BB{i}=step(FDetect,(imread(fullfile(path,fileinfo(i).name)))); 
    disp(['BB :   ' num2str(i) ]);
end;
% Find number of empty BB and index of them
c=0;
for  i = 1 : filesnumber(1,1)
   if  isempty(BB{i})
        c=c+1;
        indexempty(c)=i;
   end;
end;
% Replace the empty cells with bounding box
for  i = 1 : c
BB{indexempty(i)}=[40 60 180 180];
end;
% Removing other founded faces and keep just frist face or box
for  i = 1 : filesnumber(1,1)
    BB{i}=BB{i}(1,:);
end;
% Croping the Bounding box(face)
for i = 1 : filesnumber(1,1)
croped{i}=imcrop(images{i},BB{i}); 
    disp(['Croped :   ' num2str(i) ]);
end;
%rgb to gray convertion
for i = 1 : filesnumber(1,1)
hist{i}=rgb2gray(croped{i});
    disp(['To Gray :   ' num2str(i) ]);
end;
%imadjust
for i = 1 : filesnumber(1,1)
adjusted{i}=imadjust(hist{i}); 
    disp(['Image Adjust :   ' num2str(i) ]);
end;
% Resize the final image size
for i = 1 : filesnumber(1,1)
resized{i}=imresize(croped{i}, [128 128]); 
    disp(['Image Resized :   ' num2str(i) ]);
end;

%% Detecting Eyes
EyePairBigDetect = vision.CascadeObjectDetector('EyePairBig','MergeThreshold',16);
% Returns Bounding Box values based on number of objects
for i = 1 : filesnumber(1,1)
BBeye{i}=step(EyePairBigDetect,resized{i}); 
    disp(['BB :   ' num2str(i) ]);
end;
% Find number of empty BB and index of them
c=0;
for  i = 1 : filesnumber(1,1)
   if  isempty(BBeye{i})
        c=c+1;
        indexemptyeye(c)=i;
   end;
end;
% Replace the empty cells with bounding box
for  i = 1 : c
BBeye{indexemptyeye(i)}=[27 43 72 18];
end;
% Removing other founded eyes and keep just frist face or box
for  i = 1 : filesnumber(1,1)
    BBeye{i}=BBeye{i}(1,:);
end;
% Removing eyes and keeping the rest of the face
FinalFace=resized;
for i = 1 : filesnumber(1,1)
one=BBeye{i}(1,1);
two=BBeye{i}(1,2);
three=BBeye{i}(1,3);
four=BBeye{i}(1,4);
FinalFace{i}(two:two+four, one:one+three, :) = 255; 
disp(['Extracting Eyes From Face :   ' num2str(i) ]);
end;
% Resize the final image size
for i = 1 : filesnumber(1,1)
resizedeye{i}=imresize(FinalFace{i}, [128 128]); 
GrayNoEye{i}=rgb2gray(resizedeye{i});
    disp(['Image Resize and Gray Conversion :   ' num2str(i) ]);
end;
% Montage plot
montage(resized); title('Originals');
figure;
montage(GrayNoEye); title('Cropped');
% Save to disk
fsize=filesnumber(1,1);
for i = 1:fsize   
   imwrite(GrayNoEye{i},strcat('New',num2str(i),'.jpg'));
end