
module Datastructures
export ProcessingParameters, ImagingGrid, ImagingParameters

mutable struct ProcessingParameters
    precision::DataType
    ContrastFunction::Function
    Greyscaler::Function
    blackpoint::Array{UInt16,1}
    path::String
    save_path::String
    width::Int64
    height::Int64
    debug::Bool
end 

mutable struct ImagingGrid
    x::Array{Int64,1}
    y::Array{Int64,1}
    z::Array{Int64,1}
    exp::Array{String,1}
end


struct ImagingParameters
    exposure::String
    magnification::Float64
    px_size::Float64
    steps_per_mm::Float64
    overlap::Float64
    path::String
    save_path::String
    bps::Int64
    im_width::Int64
    im_height::Int64
    offset::Vector{Int64}
end

end # module