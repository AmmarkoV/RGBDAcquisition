#include "acquisitionScriptInput.h"
#include <stdio.h>
#include <stdlib.h>

#include "../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/TrajectoryParser/InputParser_C.h"

int executeScriptFromFile(ModuleIdentifier moduleID,DeviceIdentifier devID,const char * filename)
{
  char * line = NULL;
  size_t len = 0;
  ssize_t read;

  FILE * fp = fopen(filename,"r");
  if (fp!=0)
  {
    struct InputParserC * ipc = InputParser_Create(2048,4);

    char * line = NULL;
    size_t len = 0;

    while ((read = getline(&line, &len, fp)) != -1)
    {

    }

    InputParser_Destroy(ipc);
    fclose(fp);
    if (line) { free(line); }
    return 1;
  }
 return 0;
}
