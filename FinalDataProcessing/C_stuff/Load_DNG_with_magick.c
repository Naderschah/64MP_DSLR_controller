#include <stdio.h>
#include <stdlib.h>
#include <MagickWand/MagickWand.h>
/*
This doesnt work, some dependency for loading dngs is missing so libraw instead
*/

#define ThrowWandException(wand) \
{ \
  char \
    *description; \
 \
  ExceptionType \
    severity; \
 \
  description=MagickGetException(wand,&severity); \
  (void) fprintf(stderr,"%s %s %lu %s\n",GetMagickModule(),description); \
  description=(char *) MagickRelinquishMemory(description); \
  exit(-1); \
}

Image* LoadDNG(char *path)
{
    printf("Loading %s\n", path);
    MagickBooleanType status;
    MagickWand *magick_wand;
    // Set ip
    MagickWandGenesis();
    magick_wand=NewMagickWand();
    printf("Created Wand\n");
    // Read Image and check it worked
    status=MagickReadImage(magick_wand,path);
    if (status == MagickFalse)
        ThrowWandException(magick_wand);
    printf("Read File");
    size_t width = MagickGetImageWidth(magick_wand);
    size_t height = MagickGetImageHeight(magick_wand);
    // Create char array to hold the data
    Image* data = GetImageFromMagickWand(magick_wand);
    return data;
}

int main(int argc,char **argv)
{   
    Image* data = LoadDNG(argv[1]);
    free(data);
    MagickWandTerminus();
}