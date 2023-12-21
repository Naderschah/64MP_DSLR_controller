#include <stdio.h>
#include <stdlib.h>
#include <libraw.h>


/*
Functions to load img data in julia
Compiled like:
gcc -o LibRaw_LoadDNG.o -c LibRaw_LoadDNG.c `pkg-config --cflags --libs libraw` -fPIC // Last flag suggested by compiler 
gcc -shared -o LibRaw_LoadDNG.so LibRaw_LoadDNG.o -lm -fPIC `pkg-config --cflags --libs libraw`
*/



extern void LoadDNG(uint16_t* array, char *path, int *success)
{
    // Pointer to some structure all libraw functions require
    printf("    Opening %s\n", path);
    *success = 1;
    libraw_data_t *libraw_data_t_ptr =libraw_init(0);
    // Set preprocessing 
    libraw_data_t_ptr->params.user_flip = 6;// 5=90CCW, 6=90CW
    libraw_data_t_ptr->params.output_bps = 16;
    libraw_data_t_ptr->params.use_auto_wb = 0;
    libraw_data_t_ptr->params.use_camera_wb = 0;
    libraw_data_t_ptr->params.output_color = 0; // raw
    libraw_data_t_ptr->params.user_flip = 1; // raw
    int zero = 0;
    int one = 1;
    // Load File
    if (libraw_open_file(libraw_data_t_ptr, path)!=0) // Anything other than 0 bad
    {
        fprintf(stderr, "\n     Error Opening File %s\n",path);
        fprintf(stderr, "%s", libraw_strerror(libraw_open_file(libraw_data_t_ptr, path)));
        libraw_close(libraw_data_t_ptr);
        *success = 0;
        return;
    }
    // unpack
    if (libraw_unpack(libraw_data_t_ptr)!=0) 
    {
        fprintf(stderr, "\n     Error Preprocessing File %s\n",path);
        libraw_close(libraw_data_t_ptr);
        *success = 0;
        return;
    }

    // Preprocess image 
    libraw_dcraw_process(libraw_data_t_ptr);// Argument specifies error code 
    // Allocate RGB bitmap
    libraw_processed_image_t *image = libraw_dcraw_make_mem_image(libraw_data_t_ptr,NULL); //return must be freed seperately
    // Free all except for var image
    libraw_recycle(libraw_data_t_ptr);
    // Grab data free data struct 
    int size = image->height * image->width * image->colors;
    for (int i=0; i < size; i++)
    {
        array[i] = *((uint16_t*)&(image->data)+i); // Reassign elements to julia array
    }
    libraw_dcraw_clear_mem(image);
    return;
}

// Testing
//int main(int argc,char **argv)
//{   
//    libraw_processed_image_t *img = LoadDNG("/home/felix/rapid_storage_1/img_6/20800_57012_50712_exp196608mus.dng");
//    printf("Data size: %u\n", img->data_size);
//    printf("Format: %d\n", (int)(img->type));
//    FreeMemory(img);
//
//}