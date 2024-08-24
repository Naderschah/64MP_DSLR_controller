"""
This module contains code for post processing raw images


Aimed at:
- CCM
- Gamma curve


First segment corresponds to CCM's and gamma curve taken from scientific image
"""
module PostProcessing
CCM_7600 = [2.205529206931895, -1.1826662383072108, 0.0032019529917605167,-0.122572009780486, 1.6037258133595753, -0.5073973734282445,0.0020132587619863425, -0.4556590236414181, 1.471939788496745]
CCM_7600 = reshape(CCM_7600, 3, 3)
CCM_6600 = [2.1424988751299714, -1.134760232367728, 0.010730356010435522,-0.14021846798466234, 1.600822462230719, -0.48379204794526487,0.015521315410496622, -0.49463630325832275, 1.4933313534840327]
CCM_6600 = reshape(CCM_6600, 3, 3)
CCM_5600 = [2.062652337917251, -1.0658386679125478, 0.011886354256281267,-0.16319197721451495, 1.598363237584736, -0.45422061523742235,0.03465810928795378, -0.5535454108047286, 1.5269025836946852]
CCM_5600 = reshape(CCM_5600, 3, 3)
CCM_5100 = [2.014680536961399, -1.0195930302148566, 0.007728256612638915,-0.17751999660735496, 1.5977081555831, -0.4366085498741474,0.04741267583041334, -0.5950327902073489, 1.5512919847321853]
CCM_5100 = reshape(CCM_5100, 3, 3)
CCM_4600 = [1.960355191764125, -0.9624344812121991, -0.0017122408632169205,-0.19444620905212898, 1.5978493736948447, -0.416727638296156,0.06310261513271084, -0.6483790952487849, 1.5834605477213093]
CCM_4600 = reshape(CCM_4600, 3, 3)
CCM_2000 = [1.5813882365848004, -0.35293683714581114, -0.27378771561617715,-0.4347297185453639, 1.5792631087746074, -0.12102601986382337,0.2322290578987574, -1.4382672640468128, 2.1386425781770755]
CCM_2000 = reshape(CCM_2000, 3, 3)
gamma = [0, 0,512, 2304,1024, 4608,1536, 6573,2048, 8401,2560, 9992,3072, 11418,3584, 12719,4096, 13922,4608, 15045,5120, 16103,5632, 17104,6144, 18056,6656, 18967,7168, 19839,7680, 20679,8192, 21488,9216, 23028,10240, 24477,11264, 25849,12288, 27154,13312, 28401,14336, 29597,15360, 30747,16384, 31856,17408, 32928,18432, 33966,19456, 34973,20480, 35952,22528, 37832,24576, 39621,26624, 41330,28672, 42969,30720, 44545,32768, 46065,34816, 47534,36864, 48956,38912, 50336,40960, 51677,43008, 52982,45056, 54253,47104, 55493,49152, 56704,51200, 57888,53248, 59046,55296, 60181,57344, 61292,59392, 62382,61440, 63452,63488, 64503,65535, 65535]

function apply_gamma_correction(img, gamma_curve=gamma)
    """
    img: -> Img data
    gamma_curve -> flat lsit of form x,y,x,y,... (like picamera config file)
    """
    x = gamma_curve[1:2:end]  
    y = gamma_curve[2:2:end]  
    lut = zeros(UInt16, maximum(x) + 1)  # Lookup table for speed
    for i in 1:length(x)-1
        for j in x[i]:x[i+1]
            lut[j+1] = round(UInt16, y[i] + (y[i+1] - y[i]) * ((j - x[i]) / (x[i+1] - x[i])))
        end
    end
    return map(pixel -> lut[pixel + 1], img)
end


function apply_ccm(img, ccm=CCM_2000) # ~5000 K
    # Flatten the 3x3 CCM into a linear transformation
    f(x) = clamp.(ccm * x, 0, 1)
    return mapslices(f, img, dims=3)
end

end #module