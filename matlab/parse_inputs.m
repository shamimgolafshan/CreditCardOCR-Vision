function is3D = parse_inputs(X)

is3D = (ndims(X) == 3);
if is3D
    % RGB
    if (size(X,3) ~= 3)
        error(message('MATLAB:images:rgb2gray:invalidInputSizeRGB'))
    end
    % RGB can be single, double, int8, uint8,
    % int16, uint16, int32, uint32, int64 or uint64
    validateattributes(X, {'numeric'}, {}, mfilename, 'RGB');
elseif ismatrix(X)
    % MAP
    if (size(X,2) ~= 3 || size(X,1) < 1)
        error(message('MATLAB:images:rgb2gray:invalidSizeForColormap'))
    end
    % MAP must be double
    if ~isa(X,'double')
        error(message('MATLAB:images:rgb2gray:notAValidColormap'))
    end
else
    error(message('MATLAB:images:rgb2gray:invalidInputSize'))
end