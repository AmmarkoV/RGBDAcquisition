#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
 

int main(int argc, char *argv[])
{
  unsigned int X21sign,Y21sign,Z21sign,X23sign,Y23sign,Z23sign; 

  unsigned int hashedResponse = 0;


  fprintf(stderr,"enum hashesBruteShorthand\n");
  fprintf(stderr,"{\n"); 


         for (Z23sign=0; Z23sign<3; Z23sign++)
          {

      for (Y23sign=0; Y23sign<3; Y23sign++)
       {


   for (X23sign=0; X23sign<3; X23sign++)
     {


         for (Z21sign=0; Z21sign<3; Z21sign++)
          {

      for (Y21sign=0; Y21sign<3; Y21sign++)
       {

   for (X21sign=0; X21sign<3; X21sign++)
     {



               hashedResponse = X21sign + (3 * Y21sign) + (3*3 * Z21sign) + (3*3*3 * X23sign) + (3*3*3*3 * Y23sign) + (3*3*3*3*3 * Z23sign);

               if (X21sign==0) { fprintf(stderr,"LT"); } else 
               if (X21sign==1) { fprintf(stderr,"EQ"); } else 
               if (X21sign==2) { fprintf(stderr,"GT"); }  

               fprintf(stderr,"_"); 

               if (Y21sign==0) { fprintf(stderr,"LT"); } else 
               if (Y21sign==1) { fprintf(stderr,"EQ"); } else 
               if (Y21sign==2) { fprintf(stderr,"GT"); }  

               fprintf(stderr,"_"); 

               if (Z21sign==0) { fprintf(stderr,"LT"); } else 
               if (Z21sign==1) { fprintf(stderr,"EQ"); } else 
               if (Z21sign==2) { fprintf(stderr,"GT"); }  

               fprintf(stderr,"_"); 
 
               if (X23sign==0) { fprintf(stderr,"LT"); } else 
               if (X23sign==1) { fprintf(stderr,"EQ"); } else 
               if (X23sign==2) { fprintf(stderr,"GT"); }  

               fprintf(stderr,"_"); 

               if (Y23sign==0) { fprintf(stderr,"LT"); } else 
               if (Y23sign==1) { fprintf(stderr,"EQ"); } else 
               if (Y23sign==2) { fprintf(stderr,"GT"); }  

               fprintf(stderr,"_"); 

               if (Z23sign==0) { fprintf(stderr,"LT"); } else 
               if (Z23sign==1) { fprintf(stderr,"EQ"); } else 
               if (Z23sign==2) { fprintf(stderr,"GT"); }  


               fprintf(stderr,"=%u,\n",hashedResponse); 

          }           
       }
     }


          }           
       }
     }


  fprintf(stderr,"//----------------------------\n"); 
  fprintf(stderr,"END_OF_ALL_THE_HASHED_CASES\n"); 
  fprintf(stderr,"};\n"); 


 return 0;
}
