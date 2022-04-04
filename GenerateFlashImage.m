function returnedImage = GenerateFlashImage(rgb, depth, colorV, flash, eyeCorrection)
    colorV = colorV/norm(colorV);
    redChannel = rgb(:,:,1)*colorV(1); % Red channel
    greenChannel = rgb(:,:,2)*colorV(2); % Green channel
    blueChannel = rgb(:,:,3)*colorV(3); % Blue channel
    dotProd = redChannel + greenChannel + blueChannel;
    dotProd = double(dotProd);
    
    red = zeros(size(rgb, 1), size(rgb, 2), 'double'); %project onto colorV
    red(:,:) = colorV(1);
    red = red.*dotProd;
    
    green = zeros(size(rgb, 1), size(rgb, 2), 'double');
    green(:,:) = colorV(2);
    green = green.*dotProd;
    
    blue = zeros(size(rgb, 1), size(rgb, 2), 'double');
    blue(:,:) = colorV(3);
    blue = blue.*dotProd;
    
    out = cat(3, red, green, blue)*eyeCorrection;
    
    if flash
        depthCorrection = (depth.*depth)/(1.5*1.5); %assuming full red channel information at 1.5m.
        out = (out)./depthCorrection; %intensity drops off with r^2.
    
    end
    
    returnedImage = uint8(out);
    
    