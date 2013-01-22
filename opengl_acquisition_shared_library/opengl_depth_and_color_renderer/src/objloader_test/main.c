#include <stdio.h>
#include <stdlib.h>

#include "../model_loader_obj.h"

char whattotest[512]={"spatoula"};

int main()
{
     struct OBJ_Model * spatoula = loadObj(whattotest);


     printf("Hello world!\n");


     unloadObj(spatoula);
    return 0;
}
