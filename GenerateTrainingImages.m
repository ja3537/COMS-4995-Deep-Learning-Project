colorV_660 = [255,0,0];
colorV_640 = [255,33,0];
colorV_380 = [97,0,97];
LE_660 = 0.000313; %scotopic luminous efficiency of the human eye
LE_640 = 0.001497;
LE_380 = 0.000589;
EC_660 = LE_660/LE_660; %correction factor for eye sensitivity
EC_640 = LE_660/LE_640;
EC_380 = LE_660/LE_380;




for index = 1:1449
    rgbImage = images(:,:,:,index);
    depthImage = depths(:,:,index);
    imwrite(rgbImage, sprintf("training/training_%d_rgb.png", index));
    
    
    img_660 = GenerateFlashImage(rgbImage, depthImage, colorV_660, true, EC_660);
    imwrite(img_660, sprintf("training/training_%d_660_f.png", index));
    
    img_640 = GenerateFlashImage(rgbImage, depthImage, colorV_640, true, EC_640);
    imwrite(img_640, sprintf("training/training_%d_640_f.png", index));
    
    img_380 = GenerateFlashImage(rgbImage, depthImage, colorV_380, true, EC_380);
    imwrite(img_380, sprintf("training/training_%d_380_f.png", index));
end
    
    
    
    